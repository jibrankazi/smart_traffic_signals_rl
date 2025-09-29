# Interpretability Analysis for Traffic RL

Deep reinforcement learning controllers for traffic signals can be
highly effective but are often opaque to engineers and policy makers.
Transparency and interpretability are especially important in
safety‑critical domains like urban traffic control【705295093246900†L186-L242】.  This document
outlines approaches to gain insight into the learned policies and
recommends how to integrate interpretability into your project.

## Why Interpretability?

* **Trust and accountability** – stakeholders need to understand why
  the controller makes certain decisions to ensure they align with
  safety and fairness requirements.
* **Debugging** – examining which features drive the policy helps
  diagnose unexpected or undesirable behaviour.
* **Scientific insight** – interpretable analyses can reveal rules or
  heuristics that generalise beyond the particular training setup.

## Methods

1. **SHAP values for state features**

   SHAP (SHapley Additive exPlanations) values provide a unified way to
   attribute the agent’s Q‑values or policy outputs to individual state
   variables.  You can treat the neural network policy as a function
   mapping state features to Q‑values and compute SHAP values for
   representative states【439831869724999†L93-L107】.  This will tell you which
   traffic features (e.g. queue length, waiting time, phase) most
   influence the agent’s decisions.

   * Collect a set of states sampled from episodes.
   * Use a SHAP library (e.g. [shap](https://github.com/slundberg/shap))
     with `KernelExplainer` or `DeepExplainer` to compute SHAP values
     for the Q‑value outputs.
   * Visualise the feature importances using bar plots or bee swarm plots.

2. **Q‑Value Heatmaps**

   For tabular or low‑dimensional Q‑learning, you can visualise the
   learned Q‑table as a heatmap across different state variables.  For
   example, plot Q‑values for each phase against queue lengths on
   approaches.  This helps understand how the agent balances traffic.

3. **Policy Interpretation via Simplified Models**

   Train a simpler, interpretable model (e.g. decision tree or linear
   model) to imitate the RL policy on the collected state–action pairs.
   This so‑called policy distillation can produce a set of rules that
   approximate the RL policy, providing human‑readable guidance on how
   the agent decides【338900643216462†L18-L34】.  Even if the distilled model
   performs slightly worse, it offers valuable insight.

## Recommended Steps

1. Log state features and actions during evaluation episodes.
2. Compute SHAP values for the trained agent using the logged data.
3. Generate a summary plot showing top features influencing decisions.
4. For a small network, visualise the Q‑table or state–value function
   as heatmaps.
5. Optionally, train a decision tree on the dataset of logged
   state–action pairs and compare its performance with the RL agent.

Including these interpretability analyses in your final report will
demonstrate a thoughtful, responsible approach to deploying RL in
real‑world settings.
