"""
q_learning_agent.py
--------------------

This module implements a simple tabular Q-learning algorithm for the
`TrafficEnv` defined in ``traffic_env.py``. The agent maintains a Q-table
mapping discrete environment states to action values. Because the state
space (queue lengths) is unbounded in theory, we discretise it by
clipping queue lengths to a maximum threshold. This keeps the Q-table
manageable and allows the algorithm to learn a reasonable policy.

Key functions:

* ``discretise_state`` – maps the continuous queue lengths to a discrete
  representation by capping queues at a specified maximum.
* ``train_q_learning`` – runs the Q-learning algorithm for a number of
  episodes, updating the Q-table based on observed transitions.

The training function returns both the learned Q-table and a list of
metrics (e.g. average queue lengths) recorded across episodes so that
progress can be analysed and visualised.
"""

from typing import Dict, Tuple, List
import numpy as np

# Import the environment without using a relative import. When this module is
# imported from a notebook or script with `src` on the Python path,
# ``traffic_env`` resolves correctly. Relative imports require package
# semantics, which are not available in this context.
from traffic_env import TrafficEnv


def discretise_state(state: Tuple[int, int], max_queue: int) -> Tuple[int, int]:
    """Discretise the environment state by capping queue lengths.

    Parameters
    ----------
    state : Tuple[int, int]
        The continuous queue lengths (ns_queue, ew_queue).
    max_queue : int
        Maximum queue length to represent. Any value above this is
        clipped to max_queue.

    Returns
    -------
    Tuple[int, int]
        The discretised state with queue lengths clipped.
    """
    ns, ew = state
    return (min(ns, max_queue), min(ew, max_queue))


def train_q_learning(
    env: TrafficEnv,
    episodes: int = 200,
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_queue: int = 10,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[float]]:
    """Train a Q-learning agent on the provided environment.

    Parameters
    ----------
    env : TrafficEnv
        The environment to train on.
    episodes : int
        Number of training episodes.
    gamma : float
        Discount factor for future rewards.
    alpha : float
        Learning rate for Q-table updates.
    epsilon : float
        Epsilon-greedy exploration rate. With probability epsilon the agent
        chooses a random action; otherwise it chooses the best-known action.
    max_queue : int
        Maximum queue length to discretise states. Higher values allow
        more granularity but increase the Q-table size.

    Returns
    -------
    q_table : Dict[Tuple[int, int], np.ndarray]
        The learned Q-table mapping discretised states to action values.
    avg_total_queues : List[float]
        A list containing the average total queue length per episode. Useful
        for monitoring learning progress.
    """
    # Initialise Q-table as a dictionary mapping state -> action value array
    q_table: Dict[Tuple[int, int], np.ndarray] = {}
    avg_total_queues: List[float] = []

    for ep in range(episodes):
        state = env.reset()
        disc_state = discretise_state(state, max_queue)
        total_queue = 0

        done = False
        while not done:
            # Initialise row in Q-table if unseen
            if disc_state not in q_table:
                q_table[disc_state] = np.zeros(2)  # two actions

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.sample_action()
            else:
                # Choose the action with the highest Q-value (break ties randomly)
                best_actions = np.flatnonzero(
                    q_table[disc_state] == q_table[disc_state].max()
                )
                action = int(np.random.choice(best_actions))

            next_state, reward, done, _ = env.step(action)
            disc_next_state = discretise_state(next_state, max_queue)

            # Initialise next state in Q-table if unseen
            if disc_next_state not in q_table:
                q_table[disc_next_state] = np.zeros(2)

            # Q-learning update rule
            old_value = q_table[disc_state][action]
            next_max = q_table[disc_next_state].max()
            q_table[disc_state][action] = old_value + alpha * (
                reward + gamma * next_max - old_value
            )

            # Move to next state
            disc_state = disc_next_state
            total_queue += -(reward)  # reward is negative total queue

        avg_total_queues.append(total_queue / env.max_steps)

    return q_table, avg_total_queues