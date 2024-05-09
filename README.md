# Vehicle-to-Grid (V2G) Scheduling using Reinforcement Learning

## Project Overview
This project explores the application of Reinforcement Learning (RL) to optimize V2G scheduling within a microgrid. The key focus is on leveraging the battery storage of electric vehicles (EVs) to balance the grid's energy supply and demand effectively. We employ two primary RL techniques: Deep Q-Learning (DQL) and Actor-Critic models, to develop and compare different strategies for energy management.

## Introduction
V2G technology allows electric vehicles to interact with the power grid to either return energy to the grid or charge their batteries during non-peak times. By optimally scheduling these interactions, V2G can enhance grid stability and ensure efficient use of renewable energy sources.

## Methodology
- **Markov Decision Process (MDP)**: The problem is formulated as an MDP, considering state spaces such as demand, solar and wind power generation, and EVs' State of Charge (SoC).
- **Reinforcement Learning Models**:
  - **Deep Q-Learning**: Utilizes a neural network to approximate Q-values, helping to determine the optimal policy without the need for a model of the environment's dynamics.
  - **Actor-Critic Model**: Incorporates two neural networks acting in tandem (actor and critic) to directly estimate the policy and the value function, respectively.

## Environment Setup
- **State Space**: Includes grid demand, renewable energy output, and the charge status of connected EVs.
- **Actions**: Defined as the fraction of EVs charging or discharging at any given time.

## Results
Our experiments demonstrate that while DQL provides valuable insights into policy development, it faces challenges like high sensitivity to state space continuity and reward variability. The Actor-Critic model shows superior performance in handling continuous state spaces with minimal parameter tuning.

## Conclusion
The Actor-Critic model exhibits promising capabilities in adapting to new and variable conditions, suggesting a strong potential for future exploration and application in real-world scenarios.

## Usage
Files to set up the environment, run simulations, and evaluate models are provided in this repository.

## Data

Caiso Zone 1 Data can be found at (https://zenodo.org/records/5130612)


