"""
traffic_env.py
----------------

This module defines a simple traffic signal control environment for reinforcement
learning experiments. The environment is intentionally lightweight so it can
run without external dependencies such as `gym` or SUMO. It captures the core
aspects of a single intersection with two competing directions: north–south
and east–west. Cars arrive randomly and queue up until their direction
receives a green light. The agent's job is to choose which direction gets
the green light at each time step to minimize total queue length.

Key concepts:

* **State** – a tuple of integers `(ns_queue, ew_queue)` representing the
  number of vehicles waiting in the north–south and east–west directions.
* **Actions** – an integer `0` or `1`. Action `0` gives a green light to
  the north–south direction for one time step; action `1` gives a green
  light to the east–west direction. Only one direction can have green
  at a time.
* **Reward** – negative of the total queue length, `-(ns_queue + ew_queue)`. A
  smaller queue yields a higher (less negative) reward. This encourages
  the agent to reduce both queues over time.
* **Dynamics** – at each step, new cars arrive following a Bernoulli process
  (with potentially different rates for each direction). When a direction
  receives green, up to `depart_rate` cars leave its queue.

The environment exposes a familiar `reset()`/`step()` interface similar to
OpenAI Gym. It is deliberately simple enough to illustrate the core
reinforcement learning loop without requiring external packages.

Example
-------

```python
from traffic_env import TrafficEnv

env = TrafficEnv(max_steps=100)
state = env.reset()
done = False
while not done:
    action = 0  # always give green to north–south as a naive policy
    next_state, reward, done, info = env.step(action)
    state = next_state
```
"""

import random
from typing import Tuple, Dict


class TrafficEnv:
    """A simple traffic signal control environment."""

    def __init__(
        self,
        max_steps: int = 60,
        arrival_rate_ns: float = 0.5,
        arrival_rate_ew: float = 0.5,
        depart_rate: int = 2,
        seed: int = 0,
    ) -> None:
        """
        Initialise the environment.

        Parameters
        ----------
        max_steps : int
            Number of time steps per episode.
        arrival_rate_ns : float
            Probability of a car arriving per step in the north–south direction.
        arrival_rate_ew : float
            Probability of a car arriving per step in the east–west direction.
        depart_rate : int
            Number of cars that can leave the queue when the light is green.
        seed : int
            Random seed for reproducibility.
        """
        self.max_steps = max_steps
        self.arrival_rate_ns = arrival_rate_ns
        self.arrival_rate_ew = arrival_rate_ew
        self.depart_rate = depart_rate
        self.random = random.Random(seed)
        self.steps = 0
        self.ns_queue = 0
        self.ew_queue = 0

    def reset(self) -> Tuple[int, int]:
        """Reset the environment to its initial state.

        Returns
        -------
        state : Tuple[int, int]
            The initial state `(ns_queue, ew_queue)`.
        """
        self.steps = 0
        self.ns_queue = 0
        self.ew_queue = 0
        return (self.ns_queue, self.ew_queue)

    def _arrivals(self) -> None:
        """Simulate new vehicle arrivals for both directions."""
        # Bernoulli arrival: each step, a car may arrive with probability arrival_rate
        if self.random.random() < self.arrival_rate_ns:
            self.ns_queue += 1
        if self.random.random() < self.arrival_rate_ew:
            self.ew_queue += 1

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, float]]:
        """Advance the environment by one step.

        Parameters
        ----------
        action : int
            The action taken by the agent (0 for north–south green, 1 for east–west green).

        Returns
        -------
        next_state : Tuple[int, int]
            The new state `(ns_queue, ew_queue)` after applying the action.
        reward : float
            The reward received: negative total queue length.
        done : bool
            True if the episode has terminated, False otherwise.
        info : Dict[str, float]
            Additional information about the step (e.g., current time step).
        """
        if action not in (0, 1):
            raise ValueError("Action must be 0 (NS green) or 1 (EW green)")

        # Depart vehicles on selected direction
        if action == 0:
            self.ns_queue = max(0, self.ns_queue - self.depart_rate)
        else:
            self.ew_queue = max(0, self.ew_queue - self.depart_rate)

        # New arrivals after departure
        self._arrivals()

        # Increment step counter
        self.steps += 1

        # Reward is negative total queue length (we want to minimise queues)
        reward = -(self.ns_queue + self.ew_queue)

        done = self.steps >= self.max_steps
        next_state = (self.ns_queue, self.ew_queue)
        info = {"t": self.steps}
        return next_state, reward, done, info

    @property
    def state(self) -> Tuple[int, int]:
        """Return the current state without advancing the environment."""
        return (self.ns_queue, self.ew_queue)

    def sample_action(self) -> int:
        """Return a random action (useful for exploration)."""
        return self.random.choice([0, 1])