"""
run_multi_seed_evaluation.py
--------------------------------

This script provides a template for evaluating the deep reinforcement
learning (RL) agent across multiple random seeds.  RL training can be
highly stochastic due to random initializations, exploration noise and
environment randomness.  Evaluating a single training run may give a
misleading picture of performance【520273785081046†L50-L63】.  To obtain
robust estimates and quantify uncertainty we recommend training and
evaluating the agent several times and reporting the mean and
confidence interval of performance metrics (e.g. average delay or
waiting time).  This script illustrates how such an evaluation could
be structured.

Usage
-----
The functions provided here assume you have a callable `train_agent`
function (e.g. in `src/train_agent.py`) that accepts a random seed
and returns evaluation metrics for the trained agent on a fixed test
set.  Similarly, `evaluate_baseline` should evaluate a baseline policy
such as a fixed‐time or actuated controller.  Replace these
placeholders with calls to your actual training/evaluation code.

The script runs the training across `N_SEEDS` different seeds,
collects the performance metric for each run, and then computes the
mean and 95 % confidence interval.  Optionally, it performs a
paired t‐test to assess whether the agent’s performance is
significantly better than a baseline【520273785081046†L50-L63】.
"""

import numpy as np
from typing import List, Tuple
import scipy.stats as stats


def train_agent(seed: int) -> float:
    """Placeholder for training your RL agent with a specific seed.

    Args:
        seed: Random seed used for reproducibility.

    Returns:
        float: The performance metric (e.g. average waiting time) on a
        test set.  Replace this with your own implementation.
    """
    # TODO: Import your training code and call it here.  For example:
    # from src.train_agent import train_and_evaluate
    # return train_and_evaluate(seed)
    raise NotImplementedError("Replace with call to your training code")


def evaluate_baseline() -> float:
    """Evaluate a non‑RL baseline controller (e.g. actuated or
    fixed‑time).  Returns the performance metric for the baseline.

    Replace this stub with your actual baseline evaluation.
    """
    raise NotImplementedError("Replace with call to your baseline evaluation")


def compute_mean_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute mean and half‑width of the confidence interval for a list of
    measurements.

    Args:
        values: List of performance metrics from multiple seeds.
        confidence: Desired confidence level (default 0.95 for 95 % CI).

    Returns:
        mean: The sample mean of the values.
        half_width: Half the width of the confidence interval.
    """
    arr = np.array(values)
    mean = arr.mean()
    sem = stats.sem(arr)
    # t multiplier for two‑sided CI
    n = len(arr)
    if n > 1:
        t_mult = stats.t.ppf((1 + confidence) / 2., n - 1)
        half_width = t_mult * sem
    else:
        half_width = 0.0
    return mean, half_width


def main() -> None:
    N_SEEDS = 5  # Adjust number of seeds as needed

    # Evaluate baseline once (deterministic or averaged over seeds)
    try:
        baseline_performance = evaluate_baseline()
    except NotImplementedError:
        print("Please implement evaluate_baseline() before running.")
        return

    performances: List[float] = []
    for seed in range(N_SEEDS):
        try:
            perf = train_agent(seed)
        except NotImplementedError:
            print("Please implement train_agent() before running.")
            return
        performances.append(perf)
        print(f"Seed {seed}: performance = {perf:.3f}")

    mean, half_width = compute_mean_ci(performances)
    print(f"\nAgent mean performance: {mean:.3f} ± {half_width:.3f} (95% CI)")
    print(f"Baseline performance:     {baseline_performance:.3f}")

    # Perform paired t‑test (agent vs baseline)
    # Here we assume baseline is constant; for a stochastic baseline
    # provide a list of baseline performances of equal length
    baseline_array = np.full(len(performances), baseline_performance)
    t_stat, p_val = stats.ttest_rel(performances, baseline_array)
    print(f"Paired t‑test: t = {t_stat:.3f}, p = {p_val:.3f}")
    if p_val < 0.05:
        print("Result is statistically significant at 5% level")
    else:
        print("Result is NOT statistically significant at 5% level")


if __name__ == "__main__":
    main()
