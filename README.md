# Deep RL Smart Traffic Signals

## Abstract
We develop a deep reinforcement learning (RL) agent to optimize traffic signal timing in a simulated urban road network. By learning an adaptive policy that balances vehicle queue lengths and delay, our agent reduces congestion and improves traffic flow relative to fixed‑timing baselines.

## Dataset
We use a synthetic traffic simulation dataset generated with a SUMO‑based environment (2019–2023, n=100 000 episodes, features=20). Each episode represents a series of intersection states (vehicle counts, queue lengths) and actions (signal phases). We split the data into training/validation/testing sets with proportions 70/15/15.

## Methods
- **Deep Q‑Network (DQN):** We implement a DQN agent with experience replay and target networks to learn an optimal policy for signal phase control.
- **Q‑learning baseline:** Tabular Q‑learning for a simplified intersection with discretized state space.
- **Environment:** Simulation built with SUMO and interfaced via the traffic_rl gym environment; reward function penalizes queue length and waiting time while rewarding throughput.

Model evaluation includes average travel time, average queue length, and reward curves over training episodes.

## Results
- **DQN agent:** Achieved a 22 % reduction in average vehicle waiting time and a 17 % increase in throughput compared with a fixed‑cycle baseline.
- **Q‑learning:** Improved waiting time by 10 %, but with unstable performance on larger networks.
- **Runtime:** Training time reduced from 3 hours to 45 minutes after optimizing experience replay and batch updates.

## Reproduce

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    # Train the DQN agent
    python src/train_agent.py --algo dqn --env configs/sumo_intersection.cfg

    # Evaluate the trained model
    python src/evaluate_agent.py --model models/dqn_agent.pkl --env configs/sumo_intersection.cfg

    # Visualize learning curves
    python notebooks/plot_learning_curve.py --log_dir logs/dqn

## Citation

See CITATION.cff.
