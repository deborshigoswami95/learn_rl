import numpy as np
import matplotlib.pyplot as plt

def _cumulative(values):
    """Return cumulative sum as a numpy array (float)."""
    if len(values) == 0:
        return np.array([])
    return np.cumsum(values).astype(float)

def _moving_average(values, window=50):
    """
    Simple moving average using a fixed window.
    Pads the left side so the result is length-equal to input for easy plotting.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([])
    window = max(1, min(window, n))
    conv = np.convolve(values, np.ones(window)/window, mode="valid")
    # Left pad so length matches original sequence
    pad = np.full(window-1, conv[0])
    return np.concatenate([pad, conv])

def plot_cumulative_reward(env):
    """
    Plot cumulative reward trajectory from env.rewards_history.
    """
    rewards = env.rewards_history
    cum_rewards = _cumulative(rewards)
    plt.figure()
    plt.plot(cum_rewards)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.grid(True)
    plt.show()

def plot_cumulative_regret(env):
    """
    Plot cumulative regret = (t * best_p) - cumulative_reward at each timestep.
    Requires env.best_p and env.rewards_history.
    """
    rewards = env.rewards_history
    cum_rewards = _cumulative(rewards)
    t = np.arange(1, len(rewards) + 1, dtype=float)
    cum_optimal = env.best_p * t
    cum_regret = cum_optimal - cum_rewards
    plt.figure()
    plt.plot(cum_regret)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Over Time")
    plt.grid(True)
    plt.show()

def plot_moving_average_reward(env, window=50):
    """
    Plot moving-average of per-step rewards to visualize short-term trends.
    """
    rewards = env.rewards_history
    ma = _moving_average(rewards, window=window)
    plt.figure()
    plt.plot(ma)
    plt.xlabel("Timestep")
    plt.ylabel(f"Moving Avg Reward (window={window})")
    plt.title("Moving-Average Reward Over Time")
    plt.grid(True)
    plt.show()

def plot_action_selection_counts(env):
    """
    Plot how many times each arm was selected.
    """
    actions = np.asarray(env.actions_history, dtype=int)
    if actions.size == 0:
        counts = np.zeros(env.num_arms, dtype=int)
    else:
        counts = np.bincount(actions, minlength=env.num_arms)
    arms = np.arange(env.num_arms)
    plt.figure()
    plt.bar(arms, counts)
    plt.xlabel("Arm")
    plt.ylabel("Selection Count")
    plt.title("Action Selection Counts")
    plt.grid(True, axis="y")
    plt.show()

def plot_bandit_diagnostics(env, window=50):
    """
    Convenience function to produce all four diagnostic plots in sequence.
    """
    plot_cumulative_reward(env)
    plot_cumulative_regret(env)
    plot_moving_average_reward(env, window=window)
    plot_action_selection_counts(env)
