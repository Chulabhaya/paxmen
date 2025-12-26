from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete
from jaxmarl.wrappers.baselines import JaxMARLWrapper, get_space_dim
from matplotlib.colors import ListedColormap

# jax.config.update("jax_disable_jit", True)

# -----------------------------------------------------------------------------
# Constants / Encoding
# -----------------------------------------------------------------------------
N_ACTIONS = 5  # 0: Up, 1: Down, 2: Right, 3: Left, 4: Eat

EMPTY = 0.0
WALL = -1.0
AGENT = 0.1  # marker in observations / world-state

REWARD_POINT = 1.0
MOVE_REWARD = -0.025

OBS_RADIUS = 2  # => 5x5 local observation window
MAX_STEPS = 100

NUM_DOTS_PER_ROOM = 3
SHORTEST_CORRIDOR_LENGTH = 4  # in grid cells; "scl"


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@struct.dataclass
class State:
    """Environment state for PaxMen JaxMARL environment.

    Attributes:
        grid: Grid representation of shape (H, W) containing wall/empty markers.
        agent_pos: Agent positions of shape (N, 2) as int32 [row, col] coordinates.
        dot_pos: Dot positions of shape (N_DOTS, 2) as int32 [row, col] coordinates.
        dot_active: Boolean array of shape (N_DOTS,) indicating which dots are active.
        step: Current timestep counter as int32.
        done: Boolean array of shape (N,) indicating which agents are done.
        layout_id: Integer ID of the current maze layout variant.
    """

    grid: chex.Array  # (H, W)
    agent_pos: chex.Array  # (N, 2) int32
    dot_pos: chex.Array  # (N_DOTS, 2) int32
    dot_active: chex.Array  # (N_DOTS,) bool
    step: jnp.int32
    done: chex.Array  # (N,) bool
    layout_id: jnp.int32


@dataclass(frozen=True)
class Rect:
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height  # exclusive

    @property
    def right(self) -> int:
        return self.left + self.width  # exclusive


@dataclass(frozen=True)
class ArmSpec:
    """One corridor + one terminal room."""

    name: str  # e.g. "N0", "E", "S1"
    direction: str  # "N","S","E","W"
    corridor: Rect
    room: Rect
    corridor_len: int  # in cells (area == len since width==1 or height==1)
    exit_offset: int  # offset along the hub edge (for doubled exits)


@dataclass(frozen=True)
class LayoutSpec:
    hub_top: int
    hub_left: int
    hub_size: int
    arms: Tuple[ArmSpec, ...]  # length == num_agents


# -----------------------------------------------------------------------------
# PaxMen environment
# -----------------------------------------------------------------------------
class PaxMen(MultiAgentEnv):
    """Cooperative dot-eating in a hub-and-corridors maze.

    Supports 4, 5, or 6 agents. Hub is always odd-sized:
        4 agents -> 3x3
        5 agents -> 5x5
        6 agents -> 7x7

    For 5 and 6 agents, additional corridors are created via *double exits*:
    two corridors leave the same side of the hub from two different doorway cells.
    Corridors are straight (no turns).

    Each agent receives the same shared reward: the sum of all individual dot rewards
    plus a small per-step penalty.
    """

    def __init__(self, num_agents: int = 4) -> None:
        super().__init__(num_agents=num_agents)
        assert num_agents in (4, 5, 6), "num_agents must be 4, 5, or 6"

        self.num_agents = int(num_agents)
        self.num_rooms = self.num_agents
        self.num_dots = self.num_rooms * NUM_DOTS_PER_ROOM

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_range = jnp.arange(self.num_agents)

        # Hub sizing (odd)
        self.hub_size = {4: 3, 5: 5, 6: 7}[self.num_agents]
        self.hub_half = self.hub_size // 2

        # Corridor/room sizing
        self.scl = SHORTEST_CORRIDOR_LENGTH
        self.room_v_h, self.room_v_w = 5, 3  # vertical terminal room dims
        self.room_h_h, self.room_h_w = 3, 5  # horizontal terminal room dims
        self.room_area = self.room_v_h * self.room_v_w  # 15
        self.corridor_len_base = (
            2 * self.scl
        )  # used for fixed world_state length accounting

        # Choose grid size large enough that even the longest corridors/rooms remain
        # at least OBS_RADIUS cells from the boundary for safe dynamic_slice.
        max_len = 3 * self.scl
        margin = OBS_RADIUS
        room_extent = 5
        # Ensure odd (odd hub_size + even terms => odd)
        self.grid_size = int(self.hub_size + 2 * max_len + 2 * (room_extent + margin))

        # Hub placement (centered)
        self.hub_center = self.grid_size // 2
        self.hub_top = self.hub_center - self.hub_half
        self.hub_left = self.hub_center - self.hub_half
        self._hub_rect = Rect(self.hub_top, self.hub_left, self.hub_size, self.hub_size)

        # World-state vector size (flattened hub + concatenated corridors/rooms)
        self.center_area = self.hub_size * self.hub_size
        self.world_state_size = self.center_area + self.num_rooms * (
            self.corridor_len_base + self.room_area
        )

        # Spaces
        self.observation_spaces = {
            a: Box(
                -1.0, 4.0, ((2 * OBS_RADIUS + 1) * (2 * OBS_RADIUS + 1),), dtype=jnp.float32
            )
            for a in self.agents
        }
        self.action_spaces = {a: Discrete(N_ACTIONS) for a in self.agents}
        if num_agents == 4:
            self.max_steps = 100
        elif num_agents == 5:
            self.max_steps = 115
        elif num_agents == 6:
            self.max_steps = 130

        # Layout library (4 variants per agent-count)
        self._layout_specs: Tuple[LayoutSpec, ...] = tuple(self._build_layout_specs())
        self.num_layouts = len(self._layout_specs)

        # Pre-build branches for lax.switch (JIT-friendly)
        self._layout_fns = [
            lambda g, s=s: self._apply_layout_spec(g, s) for s in self._layout_specs
        ]
        self._dot_respawn_fns = [
            lambda k, s=s: self._respawn_dots_from_spec(k, s)
            for s in self._layout_specs
        ]
        self._world_slice_fns = [
            lambda g, s=s: self._world_state_from_grid(g, s) for s in self._layout_specs
        ]

    # -------------------------------------------------------------------------
    # Layout construction
    # -------------------------------------------------------------------------
    def _build_layout_specs(self) -> List[LayoutSpec]:
        """Build 4 layout variants for each agent count.

        Corridor lengths are multiples of scl with:
            short = 1*scl, medium = 2*scl, long = 3*scl

        We choose length-multis so that total corridor cells sum to:
            num_rooms * (2*scl)
        matching the original world_state_size accounting.

        Returns:
            List[LayoutSpec]: A list of 4 layout specifications for the environment.
        """
        r0 = self.hub_center
        c0 = self.hub_center

        hub_top = r0 - self.hub_half
        hub_left = c0 - self.hub_half
        hub_bottom = hub_top + self.hub_size - 1
        hub_right = hub_left + self.hub_size - 1

        def mk_arm(
            name: str, direction: str, exit_offset: int, corridor_len: int
        ) -> ArmSpec:
            """Create corridor+room rectangles given hub geometry.

            Args:
                name: Name identifier for the arm (e.g., "N0", "E", "S1").
                direction: Cardinal direction ("N", "S", "E", or "W").
                exit_offset: Offset along the hub edge for the exit position.
                corridor_len: Length of the corridor in grid cells.

            Returns:
                ArmSpec: Specification for the corridor and room geometry.

            Raises:
                ValueError: If direction is not one of "N", "S", "E", or "W".
            """
            if direction == "N":
                exit_col = c0 + exit_offset
                corridor = Rect(
                    top=hub_top - corridor_len,
                    left=exit_col,
                    height=corridor_len,
                    width=1,
                )
                room = Rect(
                    top=hub_top - corridor_len - self.room_v_h,
                    left=exit_col - 1,
                    height=self.room_v_h,
                    width=self.room_v_w,
                )
            elif direction == "S":
                exit_col = c0 + exit_offset
                corridor = Rect(
                    top=hub_bottom + 1, left=exit_col, height=corridor_len, width=1
                )
                room = Rect(
                    top=hub_bottom + 1 + corridor_len,
                    left=exit_col - 1,
                    height=self.room_v_h,
                    width=self.room_v_w,
                )
            elif direction == "E":
                exit_row = r0 + exit_offset
                corridor = Rect(
                    top=exit_row, left=hub_right + 1, height=1, width=corridor_len
                )
                room = Rect(
                    top=exit_row - 1,
                    left=hub_right + 1 + corridor_len,
                    height=self.room_h_h,
                    width=self.room_h_w,
                )
            elif direction == "W":
                exit_row = r0 + exit_offset
                corridor = Rect(
                    top=exit_row,
                    left=hub_left - corridor_len,
                    height=1,
                    width=corridor_len,
                )
                room = Rect(
                    top=exit_row - 1,
                    left=hub_left - corridor_len - self.room_h_w,
                    height=self.room_h_h,
                    width=self.room_h_w,
                )
            else:
                raise ValueError(f"Unknown direction {direction}")
            return ArmSpec(
                name=name,
                direction=direction,
                corridor=corridor,
                room=room,
                corridor_len=int(corridor_len),
                exit_offset=int(exit_offset),
            )

        single = 0
        double = (-2, +2)

        L = 3 * self.scl
        M = 2 * self.scl
        S = 1 * self.scl

        specs: List[LayoutSpec] = []

        if self.num_agents == 4:
            variants = [
                ("N", {"N": L, "S": S, "E": M, "W": M}),
                ("S", {"N": S, "S": L, "E": M, "W": M}),
                ("E", {"N": M, "S": M, "E": L, "W": S}),
                ("W", {"N": M, "S": M, "E": S, "W": L}),
            ]
            for _, lens in variants:
                arms = (
                    mk_arm("N", "N", single, lens["N"]),
                    mk_arm("S", "S", single, lens["S"]),
                    mk_arm("E", "E", single, lens["E"]),
                    mk_arm("W", "W", single, lens["W"]),
                )
                specs.append(
                    LayoutSpec(
                        hub_top=hub_top,
                        hub_left=hub_left,
                        hub_size=self.hub_size,
                        arms=arms,
                    )
                )

        elif self.num_agents == 5:
            for double_side in ("N", "S", "E", "W"):
                if double_side == "N":
                    arms = (
                        mk_arm("N0", "N", double[0], L),
                        mk_arm("N1", "N", double[1], S),
                        mk_arm("S", "S", single, M),
                        mk_arm("E", "E", single, M),
                        mk_arm("W", "W", single, M),
                    )
                elif double_side == "S":
                    arms = (
                        mk_arm("S0", "S", double[0], L),
                        mk_arm("S1", "S", double[1], S),
                        mk_arm("N", "N", single, M),
                        mk_arm("E", "E", single, M),
                        mk_arm("W", "W", single, M),
                    )
                elif double_side == "E":
                    arms = (
                        mk_arm("E0", "E", double[0], L),
                        mk_arm("E1", "E", double[1], S),
                        mk_arm("N", "N", single, M),
                        mk_arm("S", "S", single, M),
                        mk_arm("W", "W", single, M),
                    )
                else:  # W
                    arms = (
                        mk_arm("W0", "W", double[0], L),
                        mk_arm("W1", "W", double[1], S),
                        mk_arm("N", "N", single, M),
                        mk_arm("S", "S", single, M),
                        mk_arm("E", "E", single, M),
                    )
                specs.append(
                    LayoutSpec(
                        hub_top=hub_top,
                        hub_left=hub_left,
                        hub_size=self.hub_size,
                        arms=arms,
                    )
                )

        else:  # 6 agents
            variants = [
                ("NS", ("N", "S")),
                ("EW", ("E", "W")),
                ("NE", ("N", "E")),
                ("SW", ("S", "W")),
            ]
            for _, (d1, d2) in variants:

                def doubled_arms(direction: str, tag: str) -> Tuple[ArmSpec, ArmSpec]:
                    return (
                        mk_arm(f"{tag}0", direction, double[0], L),
                        mk_arm(f"{tag}1", direction, double[1], M),
                    )

                arms_list: List[ArmSpec] = []
                for d in (d1, d2):
                    if d == "N":
                        arms_list += list(doubled_arms("N", "N"))
                    elif d == "S":
                        arms_list += list(doubled_arms("S", "S"))
                    elif d == "E":
                        arms_list += list(doubled_arms("E", "E"))
                    else:
                        arms_list += list(doubled_arms("W", "W"))

                doubled_set = {d1, d2}
                for direction in ("N", "S", "E", "W"):
                    if direction in doubled_set:
                        continue
                    arms_list.append(mk_arm(direction, direction, single, S))

                arms = tuple(arms_list[: self.num_agents])
                specs.append(
                    LayoutSpec(
                        hub_top=hub_top,
                        hub_left=hub_left,
                        hub_size=self.hub_size,
                        arms=arms,
                    )
                )

        assert len(specs) == 4, f"Expected 4 layout specs, got {len(specs)}"
        expected_corridor_sum = self.num_rooms * self.corridor_len_base
        for s in specs:
            assert len(s.arms) == self.num_agents
            corr_sum = sum(a.corridor_len for a in s.arms)
            assert corr_sum == expected_corridor_sum, (
                self.num_agents,
                corr_sum,
                expected_corridor_sum,
            )
        return specs

    @staticmethod
    def _fill_rect(grid: chex.Array, rect: Rect, value: float) -> chex.Array:
        return grid.at[rect.top : rect.bottom, rect.left : rect.right].set(value)

    def _apply_layout_spec(self, grid: chex.Array, spec: LayoutSpec) -> chex.Array:
        hub = Rect(spec.hub_top, spec.hub_left, spec.hub_size, spec.hub_size)
        grid = self._fill_rect(grid, hub, EMPTY)
        for arm in spec.arms:
            grid = self._fill_rect(grid, arm.corridor, EMPTY)
            grid = self._fill_rect(grid, arm.room, EMPTY)
        return grid

    # -------------------------------------------------------------------------
    # Sampling helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _rect_coords(rect: Rect) -> chex.Array:
        """Generate all coordinate pairs within a rectangle.

        Args:
            rect: Rectangle specification defining the area.

        Returns:
            chex.Array: Array of shape (N, 2) containing all (row, col) coordinates.
        """
        rs = jnp.arange(rect.top, rect.bottom, dtype=jnp.int32)
        cs = jnp.arange(rect.left, rect.right, dtype=jnp.int32)
        rr, cc = jnp.meshgrid(rs, cs, indexing="ij")
        return jnp.stack([rr.reshape(-1), cc.reshape(-1)], axis=-1)

    def _sample_unique_positions(
        self, key: chex.PRNGKey, rect: Rect, k: int
    ) -> chex.Array:
        """Sample k unique random positions from within a rectangle.

        Args:
            key: JAX random key for sampling.
            rect: Rectangle specification defining the sampling area.
            k: Number of unique positions to sample.

        Returns:
            chex.Array: Array of shape (k, 2) containing sampled (row, col) positions.
        """
        coords = self._rect_coords(rect)
        idx = jax.random.choice(key, coords.shape[0], shape=(k,), replace=False)
        return coords[idx]

    def _respawn_dots_from_spec(
        self, key: chex.PRNGKey, spec: LayoutSpec
    ) -> Tuple[chex.Array, chex.Array]:
        """Spawn dots in each room according to the layout specification.

        Args:
            key: JAX random key for sampling dot positions.
            spec: Layout specification defining room locations.

        Returns:
            Tuple containing:
                - dot_pos: Array of shape (num_dots, 2) with dot positions.
                - dot_active: Boolean array of shape (num_dots,) marking all dots as active.
        """
        keys = jax.random.split(key, self.num_rooms)
        dots = []
        for i, arm in enumerate(spec.arms):
            dots.append(
                self._sample_unique_positions(keys[i], arm.room, NUM_DOTS_PER_ROOM)
            )
        dot_pos = jnp.concatenate(dots, axis=0)
        dot_active = jnp.ones((self.num_dots,), dtype=bool)
        return dot_pos, dot_active

    @partial(jax.jit, static_argnums=0)
    def _respawn_dots(
        self, key: chex.PRNGKey, layout_id: jnp.int32
    ) -> Tuple[chex.Array, chex.Array]:
        return jax.lax.switch(layout_id, self._dot_respawn_fns, key)

    # -------------------------------------------------------------------------
    # Reset / Step
    # -------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the environment to initial state.

        Args:
            key: JAX random key for initializing the environment.

        Returns:
            Tuple containing:
                - obs: Dictionary mapping agent names to their observations.
                - state: Initial environment state.
        """
        key, key_layout, key_dots, key_agents = jax.random.split(key, 4)

        grid = jnp.full((self.grid_size, self.grid_size), WALL, dtype=jnp.float32)
        layout_id = jax.random.randint(
            key_layout, (), 0, self.num_layouts, dtype=jnp.int32
        )
        grid = jax.lax.switch(layout_id, self._layout_fns, grid)

        dot_pos, dot_active = self._respawn_dots(key_dots, layout_id)
        # Spawn agents in hub (unique positions)
        agent_pos = self._sample_unique_positions(
            key_agents, self._hub_rect, self.num_agents
        )

        state = State(
            grid=grid,
            agent_pos=agent_pos,
            dot_pos=dot_pos,
            dot_active=dot_active,
            step=jnp.int32(0),
            done=jnp.zeros((self.num_agents,), dtype=bool),
            layout_id=layout_id,
        )

        obs = self.get_obs(state)
        obs["world_state"] = self.get_world_state(state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one environment step.

        Args:
            key: JAX random key for stochastic operations.
            state: Current environment state.
            actions: Dictionary mapping agent names to their chosen actions.

        Returns:
            Tuple containing:
                - obs: Dictionary of observations for each agent.
                - next_state: Updated environment state.
                - rewards: Dictionary of rewards for each agent (shared reward).
                - dones: Dictionary of done flags for each agent.
                - info: Dictionary with additional information (empty).
        """
        dx_dy_options = jnp.array(
            [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=jnp.int32
        )
        action_vals = jnp.array([actions[a] for a in self.agents], dtype=jnp.int32)
        move_deltas = dx_dy_options[action_vals]

        next_positions = jnp.clip(state.agent_pos + move_deltas, 0, self.grid_size - 1)
        valid_moves = state.grid[next_positions[:, 0], next_positions[:, 1]] != WALL
        new_agent_pos = jnp.where(valid_moves[:, None], next_positions, state.agent_pos)

        # Eating
        is_eating = action_vals == 4
        match = jnp.all(state.dot_pos[:, None, :] == new_agent_pos[None, :, :], axis=-1)
        active_dot_mask = state.dot_active[:, None]
        valid_eats = match & is_eating[None, :] & active_dot_mask

        num_agents_eating_each_dot = jnp.sum(valid_eats, axis=1)
        num_agents_eating_each_dot_safe = jnp.where(
            num_agents_eating_each_dot == 0, 1, num_agents_eating_each_dot
        )

        dot_rewards = (
            REWARD_POINT / num_agents_eating_each_dot_safe[:, None]
        ) * valid_eats
        agent_dot_rewards = jnp.sum(dot_rewards, axis=0)

        base_rewards = jnp.full((self.num_agents,), MOVE_REWARD)
        per_agent = jnp.where(agent_dot_rewards > 0, agent_dot_rewards, base_rewards)
        global_reward = per_agent.sum()
        rewards = {a: global_reward for a in self.agents}

        eaten_dot = jnp.any(valid_eats, axis=1)
        new_dot_active = state.dot_active & ~eaten_dot

        # Respawn if all eaten
        all_eaten = jnp.all(~new_dot_active)
        key, new_dot_key = jax.random.split(key)
        new_dot_pos, new_dot_active = jax.lax.cond(
            all_eaten,
            lambda _: self._respawn_dots(new_dot_key, state.layout_id),
            lambda _: (state.dot_pos, new_dot_active),
            operand=None,
        )

        done = jnp.full((self.num_agents,), state.step >= (self.max_steps - 1))
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        next_state = state.replace(
            agent_pos=new_agent_pos,
            dot_pos=new_dot_pos,
            dot_active=new_dot_active,
            step=state.step + 1,
            done=done,
        )

        obs = self.get_obs(next_state)
        obs["world_state"] = self.get_world_state(next_state)
        info: Dict = {}
        return obs, next_state, rewards, dones, info

    # -------------------------------------------------------------------------
    # Observations / world-state
    # -------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _place_dots_on_grid(self, state: State) -> chex.Array:
        grid = state.grid

        def scatter_dot(i, g):
            r, c = state.dot_pos[i]
            return jax.lax.cond(
                state.dot_active[i],
                lambda gg: gg.at[r, c].set(1.0),
                lambda gg: gg,
                g,
            )

        return jax.lax.fori_loop(0, self.num_dots, scatter_dot, grid)

    def _world_state_from_grid(
        self, grid_with_agents: chex.Array, spec: LayoutSpec
    ) -> chex.Array:
        hub = grid_with_agents[
            spec.hub_top : spec.hub_top + spec.hub_size,
            spec.hub_left : spec.hub_left + spec.hub_size,
        ].reshape(-1)
        pieces = [hub]
        for arm in spec.arms:
            corr = grid_with_agents[
                arm.corridor.top : arm.corridor.bottom,
                arm.corridor.left : arm.corridor.right,
            ].reshape(-1)
            room = grid_with_agents[
                arm.room.top : arm.room.bottom,
                arm.room.left : arm.room.right,
            ].reshape(-1)
            pieces.append(corr)
            pieces.append(room)
        return jnp.concatenate(pieces, axis=0)

    @partial(jax.jit, static_argnums=0)
    def get_world_state(self, state: State) -> chex.Array:
        """Construct the global world state from the current environment state.

        Args:
            state: Current environment state.

        Returns:
            chex.Array: Flattened world state vector containing hub, corridors, and rooms.
        """
        grid_with_dots = self._place_dots_on_grid(state)

        def scatter_agents(i, g):
            r, c = state.agent_pos[i]
            return g.at[r, c].set(AGENT)

        grid_with_agents = jax.lax.fori_loop(
            0, self.num_agents, scatter_agents, grid_with_dots
        )
        return jax.lax.switch(state.layout_id, self._world_slice_fns, grid_with_agents)

    @partial(jax.jit, static_argnums=0)
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Local per-agent observations: a (2*OBS_RADIUS+1)^2 window around each agent.

        The observation includes walls, dots, and other agents (marked with AGENT).
        """
        grid_with_dots = self._place_dots_on_grid(state)

        @partial(jax.vmap, in_axes=[0, None, None])
        def _observation(agent_idx: int, grid: chex.Array, st: State) -> chex.Array:
            pos = st.agent_pos[agent_idx]
            top_left = pos - jnp.array([OBS_RADIUS, OBS_RADIUS], dtype=jnp.int32)

            local = jax.lax.dynamic_slice(
                grid,
                (top_left[0], top_left[1]),
                (2 * OBS_RADIUS + 1, 2 * OBS_RADIUS + 1),
            )

            # Mark other agents (fixed-shape loop; JIT-safe)
            rel_all = (st.agent_pos - top_left[None, :]).astype(jnp.int32)  # (N,2)
            H = 2 * OBS_RADIUS + 1

            def _mark(i, arr):
                rr, cc = rel_all[i]
                inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < H) & (i != agent_idx)

                def _do(a):
                    return a.at[rr, cc].set(AGENT)

                return jax.lax.cond(inb, _do, lambda a: a, arr)

            local = jax.lax.fori_loop(0, self.num_agents, _mark, local)
            return local.reshape(-1)

        obs_arr = _observation(self.agent_range, grid_with_dots, state)
        return {a: obs_arr[i] for i, a in enumerate(self.agents)}


class PaxMenCTRolloutManager(JaxMARLWrapper):
    """Rollout Manager for Centralized Training with Parameter Sharing.

    Used by JaxMARL Q-Learning Baselines. This wrapper:
    - Batchifies multiple environments (number defined by batch_size in __init__).
    - Adds a global state (obs["__all__"]) and global reward (rewards["__all__"]) to env.step returns.
    - Pads agent observations to ensure uniform length.
    - Adds one-hot encoded agent IDs to observation vectors.

    By default:
    - global_state is the concatenation of all agents' observations.
    - global_reward is the sum of all agents' rewards.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        batch_size: int,
        training_agents: List[str] = None,
        preprocess_obs: bool = True,
    ) -> None:
        """Initialize the PaxMen Centralized Training Rollout Manager.

        Args:
            env: The base multi-agent environment to wrap.
            batch_size: Number of parallel environments to run.
            training_agents: List of agent names to train. If None, uses all agents.
            preprocess_obs: Whether to preprocess observations (padding and agent IDs).
        """
        super().__init__(env)

        self.batch_size = batch_size

        # the agents to train could differ from the total trainable agents in the env (f.i. if using pretrained agents)
        # it's important to know it in order to compute properly the default global rewards and state
        self.training_agents = (
            self.agents if training_agents is None else training_agents
        )
        self.preprocess_obs = preprocess_obs

        # TOREMOVE: this is because overcooked doesn't follow other envs conventions
        if len(env.observation_spaces) == 0:
            self.observation_spaces = {
                agent: self.observation_space() for agent in self.agents
            }
        if len(env.action_spaces) == 0:
            self.action_spaces = {agent: env.action_space() for agent in self.agents}

        # batched action sampling
        self.batch_samplers = {
            agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0))
            for agent in self.agents
        }

        # assumes the observations are flattened vectors
        self.max_obs_length = max(
            list(map(lambda x: get_space_dim(x), self.observation_spaces.values()))
        )
        self.max_action_space = max(
            list(map(lambda x: get_space_dim(x), self.action_spaces.values()))
        )
        self.obs_size = self.max_obs_length
        if self.preprocess_obs:
            self.obs_size += len(self.agents)

        # agents ids
        self.agents_one_hot = {
            a: oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))
        }
        # valid actions
        self.valid_actions = {a: jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh = {
            a: jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n)))
            for a, u in self.action_spaces.items()
        }

        # custom global state and rewards for specific envs
        if "smax" in env.name.lower():
            self.global_state = lambda obs, state: obs["world_state"]
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
            self.get_valid_actions = lambda state: jax.vmap(env.get_avail_actions)(
                state
            )
        elif "overcooked" in env.name.lower():
            self.global_state = lambda obs, state: jnp.concatenate(
                [obs[agent].flatten() for agent in self.agents], axis=-1
            )
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
        elif "hanabi" in env.name.lower():
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
            self.get_valid_actions = lambda state: jax.vmap(env.get_legal_moves)(state)
        elif "paxmen" in env.name.lower():
            self.global_state = lambda obs, state: obs["world_state"]
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs_, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.tree.map(
                self._preprocess_obs,
                {agent: obs_[agent] for agent in self.agents},
                self.agents_one_hot,
            )
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs_, state, reward, done, infos = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.tree.map(
                self._preprocess_obs,
                {agent: obs_[agent] for agent in self.agents},
                self.agents_one_hot,
            )
            obs = jax.tree.map(
                lambda d, o: jnp.where(d, 0.0, o),
                {agent: done[agent] for agent in self.agents},
                obs,
            )  # ensure that the obs are 0s for done agents
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs, state):
        return jnp.concatenate([obs[agent] for agent in self.agents], axis=-1)

    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack([reward[agent] for agent in self.training_agents]).sum(axis=0)

    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](
            jax.random.split(key, self.batch_size)
        ).astype(int)

    @partial(jax.jit, static_argnums=0)
    def get_valid_actions(self, state):
        # default is to return the same valid actions one hot encoded for each env
        return {
            agent: jnp.tile(actions, self.batch_size).reshape(self.batch_size, -1)
            for agent, actions in self.valid_actions_oh.items()
        }

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [
            (0, max(0, self.max_obs_length - arr.shape[-1]))
        ]
        arr = jnp.pad(arr, pad_width, mode="constant", constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr


# A small palette for distinct categories: 0 => black, 1 => white, 2 => red, 3 => yellow
# You can reorder or expand if you like.
cmap = ListedColormap(["black", "white", "red", "yellow"])


def visualize_paxmen_state(state: State, save_path: str = None) -> None:
    """Convert JAX environment state to visualization and save or display.

    Creates a visual representation of the PaxMen environment state including:
    - Grid structure (walls and empty spaces)
    - Agent positions
    - Dot positions (food items)

    Args:
        state: The environment state to visualize. Must contain grid, agent_pos,
               dot_pos, and step attributes.
        save_path: Optional path to save the figure. If None, will display with plt.show().

    Note:
        Color coding: black=walls, white=empty, red=agents, yellow=dots.
    """
    # 1) Convert grid to NumPy
    base_grid = np.asarray(state.grid)  # shape (33,33) or so

    # 2) Make a copy to place dots & agents
    vis_grid = base_grid.copy()

    # 3) Overlay the agent positions
    #    Each row is (row, col). We'll set those cells to 'AGENT' value (0.1).
    #    Convert them to NumPy first:
    agent_pos_np = np.asarray(state.agent_pos)
    for r, c in agent_pos_np:
        vis_grid[r, c] = AGENT

    # 4) Overlay dot positions from each of the four rooms
    #    We'll place them as 'DOT' = 4.0 in the grid.
    dot_array = np.asarray(state.dot_pos)
    for r, c in dot_array:
        vis_grid[r, c] = REWARD_POINT

    # 5) Build a "display" array that maps WALL->0, EMPTY->1, AGENT->2, DOT->3
    #    or any scheme you like.
    display_grid = np.zeros_like(vis_grid, dtype=np.int32)

    # Mark walls => 0
    display_grid[vis_grid == WALL] = 0
    # Mark empty => 1
    display_grid[vis_grid == EMPTY] = 1
    # Mark agents => 2
    display_grid[np.isclose(vis_grid, AGENT)] = 2
    # Mark dots => 3
    display_grid[vis_grid == REWARD_POINT] = 3

    # 6) Save or show it
    plt.figure(figsize=(6, 6))
    plt.title(f"PaxMen Env at step={state.step}")
    plt.imshow(display_grid, origin="upper", cmap=cmap, vmin=0, vmax=3)
    plt.colorbar()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Initialize environment
    for i in range(1):
        key = jax.random.PRNGKey(i)
        env = PaxMen(num_agents=6)
        obs, state = env.reset(key)
        print(state.layout_id)

    # Collect states for visualization
    states = []
    states.append(state)

    # Run environment for 10 random steps
    for i in range(10):
        key, key_act = jax.random.split(key)
        key_act = jax.random.split(key_act, env.num_agents)

        # Generate random actions for each agent
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        # Step the environment
        obs, state, rewards, dones, _ = env.step(key, state, actions)
        states.append(state)

        if all(dones.values()):
            print("Episode finished early.")
            break

    for idx, state_to_vis in enumerate(states):
        visualize_paxmen_state(state_to_vis, save_path=f"paxmen_state_{idx}.png")
