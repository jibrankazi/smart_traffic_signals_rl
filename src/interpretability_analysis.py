"""
Interpretability analysis for the DQN-based traffic signal control agent.

This module demonstrates how to compute and visualise Shapley values for
the Q-network used in the smart_traffic_signals_rl project.  It relies on
the SHAP library to explain which features of the traffic state (e.g. queue
lengths, signal phase) most influence the agent’s decisions.  It also
provides a simple utility to plot a heatmap of the learned Q-values across
states for small grid networks.

Functions
---------
compute_shap_values(agent, states):
    Computes SHAP values for a trained agent on a batch of states.

plot_shap_summary(shap_values, feature_names):
    Plots a summary bar chart of mean absolute SHAP values per feature.

plot_q_heatmap(q_table):
    For tabular agents, plots a heatmap of the Q-table for two-dimensional
    state spaces (e.g. queue length on two approaches).

Notes
-----
This is a template only.  To run it you must install the `shap` package and
provide a trained agent with a `.predict()` or `.q_network` method.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt


def compute_shap_values(agent, states):
    """Compute SHAP values for a batch of states.

    Parameters
    ----------
    agent : object
        Trained RL agent with a `predict()` method that returns Q-values or
        a policy output for given states.
    states : np.ndarray
        Array of environment states (shape: [n_samples, n_features]).

    Returns
    -------
    np.ndarray
        SHAP values with shape (n_samples, n_features).
    """
    # Use a wrapper to make agent predictions compatible with SHAP
    def model_predict(data_as_numpy):
        return agent.predict(data_as_numpy)

    explainer = shap.KernelExplainer(model_predict, shap.sample(states, 100))
    shap_values = explainer.shap_values(states, nsamples=100)
    return shap_values


def plot_shap_summary(shap_values, feature_names):
    """Plot a SHAP summary bar chart of mean absolute values.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from `compute_shap_values()` (n_samples, n_features).
    feature_names : list[str]
        Names of the features corresponding to each column in the state.
    """
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.show()


def plot_q_heatmap(q_table):
    """Visualise a tabular Q-table as a heatmap.

    Useful for simple environments where the state space can be discretised
    into two dimensions.  For example, plotting Q-values for different
    combinations of queue lengths on two approaches.

    Parameters
    ----------
    q_table : np.ndarray
        Q-table with shape (n_states_dim1, n_states_dim2).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.imshow(q_table, cmap='viridis')
    ax.set_xlabel('State dimension 1')
    ax.set_ylabel('State dimension 2')
    fig.colorbar(c, ax=ax, label='Q-value')
    plt.show()


__all__ = [
    "compute_shap_values",
    "plot_shap_summary",
    "plot_q_heatmap",
]