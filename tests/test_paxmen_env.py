import jax
import jax.numpy as jnp
import pytest

from paxmen.paxmen_env import (
    AGENT,
    MAX_STEPS,
    MOVE_REWARD,
    REWARD_POINT,
    WALL,
    PaxMen,
)


@pytest.fixture
def env():
    """Initialize the PaxMen environment before each test."""
    return PaxMen()


@pytest.fixture
def key():
    """Return a random PRNGKey."""
    return jax.random.PRNGKey(42)


def test_reward_all_agents_eat_different_dots(env, key):
    """Ensure agents correctly eat dots and receive rewards."""
    obs, state = env.reset(key)

    # Move all agents to the first few dots
    new_agent_positions = state.dot_pos[: env.num_agents]  # Move agents to dots
    state = state.replace(agent_pos=new_agent_positions)

    # All agents select the "Eat" action (4)
    actions = {agent: 4 for agent in env.agents}

    # Step the environment
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Assertions: Dots should be inactive, each agent should initially
    # receive reward of 1. This is summed up to get 4, so the final global
    # reward each agent gets is 4 (because of 4 independently eaten dots)
    actual_individual_reward = jnp.ones(env.num_agents)
    actual_global_reward = actual_individual_reward.sum()
    actual_rewards = {a: actual_global_reward for a in env.agents}

    assert not jnp.any(state.dot_active[: env.num_agents]), (
        "Eaten dots should be inactive!"
    )
    assert all(rewards[a] == actual_rewards[a] for a in env.agents), (
        "Agents should receive positive rewards!"
    )


def test_two_agents_eating_same_dot(env, key):
    """Ensure that when multiple agents eat the same dot, they split the reward correctly."""
    obs, state = env.reset(key)

    # Place two agents at the same dot
    target_dot_pos = state.dot_pos[0]  # Select first dot
    new_agent_positions = jnp.stack([target_dot_pos] * 2 + list(state.agent_pos[2:]))
    state = state.replace(agent_pos=new_agent_positions)

    # Both agents select "Eat"
    actions = {env.agents[0]: 4, env.agents[1]: 4, env.agents[2]: 0, env.agents[3]: 0}

    # Step environment
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Manually calculate the global reward each agent should be getting (they should all get the same global reward)
    # Initially, all agents get a penalty
    actual_individual_reward = MOVE_REWARD * jnp.ones(env.num_agents)
    # If two agents eat the same dot, then they each get a split reward
    actual_individual_reward = actual_individual_reward.at[0].set(0.5)
    actual_individual_reward = actual_individual_reward.at[1].set(0.5)
    actual_global_reward = actual_individual_reward.sum()
    actual_rewards = {a: actual_global_reward for a in env.agents}

    # Make sure that the eaten dots are inactive
    assert state.dot_active[0] == False, "Eaten dots should be inactive!"
    assert all(rewards[a] == actual_rewards[a] for a in env.agents), (
        "Agents should receive positive rewards!"
    )


def test_two_agents_eating_different_dots(env, key):
    """Ensure that when multiple agents eat the same dot, they split the reward correctly."""
    obs, state = env.reset(key)

    # Place two agents at the same dot
    agent_1_target_dot_pos = state.dot_pos[0]  # Select first dot
    agent_2_target_dot_pos = state.dot_pos[1]  # Select second dot
    new_agent_positions = jnp.stack(
        [agent_1_target_dot_pos] + [agent_2_target_dot_pos] + list(state.agent_pos[2:])
    )
    state = state.replace(agent_pos=new_agent_positions)

    # Both agents select "Eat"
    actions = {env.agents[0]: 4, env.agents[1]: 4, env.agents[2]: 0, env.agents[3]: 0}

    # Step environment
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Manually calculate the global reward each agent should be getting (they should all get the same global reward)
    # Initially, all agents get a penalty
    actual_individual_reward = MOVE_REWARD * jnp.ones(env.num_agents)
    # If two agents eat the same dot, then they each get a split reward
    actual_individual_reward = actual_individual_reward.at[0].set(1.0)
    actual_individual_reward = actual_individual_reward.at[1].set(1.0)
    actual_global_reward = actual_individual_reward.sum()
    actual_rewards = {a: actual_global_reward for a in env.agents}

    # Make sure that the eaten dots are inactive
    assert state.dot_active[0] == False, "Eaten dots should be inactive!"
    assert state.dot_active[1] == False, "Eaten dots should be inactive!"
    assert all(rewards[a] == actual_rewards[a] for a in env.agents), (
        "Agents should receive positive rewards!"
    )


def test_three_agents_eating_same_dot(env, key):
    """Ensure that when multiple agents eat the same dot, they split the reward correctly."""
    obs, state = env.reset(key)

    # Place two agents at the same dot
    target_dot_pos = state.dot_pos[0]  # Select first dot
    new_agent_positions = jnp.stack([target_dot_pos] * 3 + list(state.agent_pos[3:]))
    state = state.replace(agent_pos=new_agent_positions)

    # Both agents select "Eat"
    actions = {env.agents[0]: 4, env.agents[1]: 4, env.agents[2]: 4, env.agents[3]: 0}

    # Step environment
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Manually calculate the global reward each agent should be getting (they should all get the same global reward)
    # Initially, all agents get a penalty
    actual_individual_reward = MOVE_REWARD * jnp.ones(env.num_agents)
    # If two agents eat the same dot, then they each get a split reward
    actual_individual_reward = actual_individual_reward.at[0].set(1 / 3)
    actual_individual_reward = actual_individual_reward.at[1].set(1 / 3)
    actual_individual_reward = actual_individual_reward.at[2].set(1 / 3)
    actual_global_reward = actual_individual_reward.sum()
    actual_rewards = {a: actual_global_reward for a in env.agents}

    # Make sure that the eaten dots are inactive
    assert state.dot_active[0] == False, "Eaten dots should be inactive!"
    assert all(rewards[a] == actual_rewards[a] for a in env.agents), (
        "Agents should receive positive rewards!"
    )


def test_all_agents_eating_same_dot(env, key):
    """Ensure that when multiple agents eat the same dot, they split the reward correctly."""
    obs, state = env.reset(key)

    # Place two agents at the same dot
    target_dot_pos = state.dot_pos[0]  # Select first dot
    new_agent_positions = jnp.stack([target_dot_pos] * 4)
    state = state.replace(agent_pos=new_agent_positions)

    # Both agents select "Eat"
    actions = {env.agents[0]: 4, env.agents[1]: 4, env.agents[2]: 4, env.agents[3]: 4}

    # Step environment
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Manually calculate the global reward each agent should be getting (they should all get the same global reward)
    # Initially, all agents get a penalty
    actual_individual_reward = MOVE_REWARD * jnp.ones(env.num_agents)
    # If two agents eat the same dot, then they each get a split reward
    actual_individual_reward = actual_individual_reward.at[0].set(0.25)
    actual_individual_reward = actual_individual_reward.at[1].set(0.25)
    actual_individual_reward = actual_individual_reward.at[2].set(0.25)
    actual_individual_reward = actual_individual_reward.at[3].set(0.25)
    actual_global_reward = actual_individual_reward.sum()
    actual_rewards = {a: actual_global_reward for a in env.agents}

    # Make sure that the eaten dots are inactive
    assert state.dot_active[0] == False, "Eaten dots should be inactive!"
    assert all(rewards[a] == actual_rewards[a] for a in env.agents), (
        "Agents should receive positive rewards!"
    )


def test_dot_respawn(env, key):
    """Ensure that dots respawn correctly when all are eaten."""
    obs, state = env.reset(key)

    # All agents select "Eat"
    actions = {agent: 4 for agent in env.agents}

    # Step until all dots are eaten
    inactive_dots_count = []

    for i in range(2):  # Two steps should remove 8 dots (4 per step)
        # Move agents to next set of dots
        start_idx = i * env.num_agents
        end_idx = (i + 1) * env.num_agents
        new_agent_positions = state.dot_pos[start_idx:end_idx]

        # Place agents at different dots
        state = state.replace(agent_pos=new_agent_positions)

        obs, state, rewards, dones, _ = env.step(key, state, actions)

        # Record the number of inactive dots
        inactive_dots_count.append(jnp.sum(~state.dot_active))

    # After two iterations, exactly 8 dots should be inactive
    assert inactive_dots_count[-1] == 8, (
        f"Expected 8 inactive dots, got {inactive_dots_count[-1]}"
    )

    # Run the third step, which should remove the last 4 dots and trigger respawn
    start_idx = 2 * env.num_agents
    end_idx = (2 + 1) * env.num_agents
    new_agent_positions = state.dot_pos[start_idx:end_idx]
    state = state.replace(agent_pos=new_agent_positions)
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Ensure all dots are active again
    assert jnp.all(state.dot_active), "All dots should be active after respawn!"


def test_agents_dont_eat_without_eating_action(env, key):
    """Ensure agents don’t eat dots when they are not selecting 'Eat'."""
    obs, state = env.reset(key)

    # Move agents to dots but do NOT select "Eat"
    new_agent_positions = state.dot_pos[: env.num_agents]
    state = state.replace(agent_pos=new_agent_positions)

    # All agents select "Move Right" (2) instead of "Eat" (4)
    actions = {agent: 2 for agent in env.agents}

    # Step environment
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Assertions: Dots should remain active, and rewards should be just move penalties
    assert jnp.all(state.dot_active[: env.num_agents]), "Dots should remain active!"
    assert all(rewards[a] < 0 for a in env.agents), (
        "Agents should not receive positive rewards!"
    )


def test_agents_dont_move_through_walls(env, key):
    """Ensure that agents can't move through walls."""
    obs, state = env.reset(key)

    # Attempt to move agents into walls
    actions = {agent: 0 for agent in env.agents}  # Move Up

    for i in range(3):  # Force agents to step upward several times
        # Step environment
        obs, state, rewards, dones, _ = env.step(key, state, actions)

    # Check that none of the agents moved into walls
    for agent_idx, pos in enumerate(state.agent_pos):
        assert state.grid[pos[0], pos[1]] != WALL, (
            f"Agent {agent_idx} should not move into walls!"
        )


def test_environment_ends_at_max_steps(env, key):
    """Ensure that the environment ends after MAX_STEPS steps."""
    obs, state = env.reset(key)

    for _ in range(MAX_STEPS):
        actions = {agent: 0 for agent in env.agents}  # Move up repeatedly
        obs, state, rewards, dones, _ = env.step(key, state, actions)

    assert all(dones.values()), "Environment should end after max steps!"


def test_rewards_are_correct(env, key):
    """Ensure that reward values make sense for different scenarios."""
    obs, state = env.reset(key)

    # Move agents randomly for 5 steps and track rewards
    total_rewards = {agent: 0 for agent in env.agents}

    for _ in range(5):
        actions = {agent: jax.random.randint(key, (), 0, 5) for agent in env.agents}
        obs, state, rewards, dones, _ = env.step(key, state, actions)

        for agent in env.agents:
            total_rewards[agent] += rewards[agent]

    # Ensure rewards are reasonable (not excessive, not all zero)
    for agent in env.agents:
        assert -5 < total_rewards[agent] < 5, (
            f"Unexpected total reward for {agent}: {total_rewards[agent]}"
        )


def test_dot_disappears_from_obs_after_eating(env, key):
    """Ensure that eating a dot removes it from the agent's observation window."""
    obs, state = env.reset(key)

    # Move each agent to a unique dot
    new_agent_positions = state.dot_pos[: env.num_agents]
    state = state.replace(agent_pos=new_agent_positions)

    # Record observations before eating
    obs_before = env.get_obs(state)

    # All agents select "Eat" action (4)
    actions = {agent: 4 for agent in env.agents}

    # Step environment
    obs_after, state, rewards, dones, _ = env.step(key, state, actions)

    # Check that the only difference in observations is that the dot has disappeared
    for i, agent in enumerate(env.agents):
        before = obs_before[agent]
        after = obs_after[agent]

        # Ensure they are not equal
        assert not jnp.array_equal(before, after), f"{agent}'s obs did not change!"

        # The only difference should be the removal of the reward dot (==1.0)
        diff = before - after

        # All values in diff should be 0, except possibly one (the removed dot)
        nonzero_diff_indices = jnp.nonzero(diff)
        num_nonzeros = len(nonzero_diff_indices[0])
        assert 1 <= num_nonzeros <= 2, (
            f"{agent} obs changed in unexpected number of places: {num_nonzeros}"
        )

        # The removed value should have been a reward dot
        for idx in nonzero_diff_indices[0]:
            assert (before[idx] == 1.0 and after[idx] == 0.0) or (
                before[idx] == 1.0 and after[idx] == 0.1
            ), (
                f"{agent} saw unexpected change in obs at index {idx}: before={before[idx]}, after={after[idx]}"
            )


def test_agent_does_not_get_reward_repeatedly_from_same_dot(env, key):
    obs, state = env.reset(key)
    new_agent_positions = state.dot_pos[: env.num_agents]
    state = state.replace(agent_pos=new_agent_positions)
    actions = {agent: 4 for agent in env.agents}

    # First eat should give reward
    obs, state, rewards, dones, _ = env.step(key, state, actions)
    for agent in env.agents:
        assert rewards[agent] > 0, f"{agent} did not get reward on first eat"

    # Second eat should give no reward (or penalty)
    obs, state, rewards, dones, _ = env.step(key, state, actions)
    for agent in env.agents:
        assert rewards[agent] <= 0, f"{agent} got reward again after dot was gone"


def test_agent_can_move_after_eating(env, key):
    obs, state = env.reset(key)
    new_agent_positions = state.dot_pos[: env.num_agents]
    state = state.replace(agent_pos=new_agent_positions)
    actions = {agent: 4 for agent in env.agents}
    obs, state, _, _, _ = env.step(key, state, actions)

    # Now try moving all agents one tile to the right
    actions = {agent: 2 for agent in env.agents}  # Action 2 = Right
    old_positions = state.agent_pos
    obs, state, _, _, _ = env.step(key, state, actions)

    # Check that all agents moved right (unless a wall blocked them)
    deltas = state.agent_pos - old_positions
    for i, agent in enumerate(env.agents):
        moved = not jnp.array_equal(state.agent_pos[i], old_positions[i])
        assert (
            moved or state.grid[old_positions[i][0], old_positions[i][1] + 1] == WALL
        ), f"{agent} failed to move after eating!"


def test_eating_on_empty_tile_does_not_reward(env, key):
    obs, state = env.reset(key)

    # Eat on empty tiles (no dots where agents initially spawn)
    actions = {agent: 4 for agent in env.agents}
    obs, state, rewards, dones, _ = env.step(key, state, actions)

    for agent in env.agents:
        assert rewards[agent] < 0, f"{agent} got reward for eating nothing!"


def test_world_state_generation(env, key):
    """Test that the world state is generated correctly."""
    obs, state = env.reset(key)

    # Get the world state from observations
    world_state = obs["world_state"]

    # 1. Check that the world state has the expected size
    expected_size = env.world_state_size
    assert world_state.shape[0] == expected_size, (
        f"World state size mismatch! Expected {expected_size}, got {world_state.shape[0]}"
    )

    # 2. Check that the world state contains valid values
    # Valid values are: WALL (-1.0), EMPTY (0.0), AGENT (0.1), REWARD_POINT (1.0)
    valid_values = jnp.array([WALL, 0.0, AGENT, REWARD_POINT])
    for val in jnp.unique(world_state):
        assert jnp.any(jnp.isclose(val, valid_values)), (
            f"World state contains invalid value: {val}"
        )

    # 3. Check that the world state contains the correct number of agents
    num_agents_in_world_state = jnp.sum(jnp.isclose(world_state, AGENT))
    assert num_agents_in_world_state == env.num_agents, (
        f"Expected {env.num_agents} agents in world state, found {num_agents_in_world_state}"
    )

    # 4. Check that the world state contains the correct number of active dots
    num_active_dots = jnp.sum(state.dot_active)
    num_dots_in_world_state = jnp.sum(jnp.isclose(world_state, REWARD_POINT))
    assert num_dots_in_world_state == num_active_dots, (
        f"Expected {num_active_dots} dots in world state, found {num_dots_in_world_state}"
    )

    # 5. Check that the world state does not contain walls (only hub, corridors, and rooms should be included)
    # The world state should only contain navigable areas
    num_walls_in_world_state = jnp.sum(jnp.isclose(world_state, WALL))
    assert num_walls_in_world_state == 0, (
        f"World state should not contain walls! Found {num_walls_in_world_state} wall cells"
    )

    # 6. Test that the world state updates correctly after agent movement
    # Move agents to dots
    new_agent_positions = state.dot_pos[: env.num_agents]
    state = state.replace(agent_pos=new_agent_positions)

    # Eat the dots
    actions = {agent: 4 for agent in env.agents}
    obs_after, state_after, _, _, _ = env.step(key, state, actions)
    world_state_after = obs_after["world_state"]

    # Check that the number of dots decreased
    num_dots_after = jnp.sum(jnp.isclose(world_state_after, REWARD_POINT))
    num_active_dots_after = jnp.sum(state_after.dot_active)
    assert num_dots_after == num_active_dots_after, (
        f"After eating, expected {num_active_dots_after} dots in world state, found {num_dots_after}"
    )

    # Check that the world state changed
    assert not jnp.array_equal(world_state, world_state_after), (
        "World state should change after agents move and eat dots!"
    )

    # 7. Check that the world state size remains constant
    assert world_state_after.shape[0] == expected_size, (
        f"World state size changed! Expected {expected_size}, got {world_state_after.shape[0]}"
    )


if __name__ == "__main__":
    pytest.main()
