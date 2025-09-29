"""
baselines.py
-------------

This module contains placeholder implementations for traditional
traffic signal control strategies that serve as baselines for your
deep RL agent.  Comparing your RL agent against strong, domain‑relevant
baselines is essential for credible evaluation【428212373104974†L55-L63】.

The two baselines suggested here are:

* **Actuated Control** – a simple feedback controller that extends or
  shortens the green phase based on real‑time vehicle presence on
  detectors.  This is commonly deployed in real traffic systems and is
  stronger than a fixed‑cycle controller.

* **Max‑Pressure Control** – a well‑studied heuristic that chooses
  signal phases to minimise network queue lengths by maximising the
  difference (pressure) between upstream and downstream queues.  It
  has been shown to perform well in congestion scenarios.

Replace the stub functions below with your actual implementations or
interfaces to the simulator.
"""

def actuated_control_step(state):
    """Compute the next action under an actuated signal controller.

    Args:
        state: The current environment state containing queue lengths
            and detector activations.

    Returns:
        action: The selected signal phase index.

    TODO: Implement a simple actuated control logic based on the
    presence of vehicles at the stop line.  For example, extend the
    green time on approaches with vehicles and switch when no vehicle is
    detected or a maximum green has elapsed.
    """
    raise NotImplementedError("Implement actuated control logic")


def max_pressure_control_step(state):
    """Compute the next action under a max‑pressure controller.

    Args:
        state: The current environment state containing queue lengths
            on all approaches and turns.

    Returns:
        action: The signal phase index that maximises network pressure.

    TODO: Implement the max‑pressure control algorithm.  Pressure for
    each phase can be computed as the sum of differences between queue
    lengths on incoming and outgoing links for movements allowed by
    that phase.  Select the phase with the maximum pressure.
    """
    raise NotImplementedError("Implement max‑pressure control logic")


def evaluate_baseline(controller_step_fn, env, episodes=10):
    """Evaluate a baseline controller over multiple episodes.

    Args:
        controller_step_fn: Function mapping state to action (e.g.
            actuated_control_step).
        env: The traffic simulation environment (e.g. SUMO or custom
            environment) with the same API used for the RL agent.
        episodes: Number of episodes to average over.

    Returns:
        float: Mean performance metric (e.g. average waiting time) over
        episodes.

    TODO: Replace this stub with your actual evaluation loop.  For
    example, reset the environment, run until done, collect rewards or
    delays, and compute the average.  Use consistent metrics to compare
    with the RL agent.
    """
    raise NotImplementedError("Implement baseline evaluation loop")
