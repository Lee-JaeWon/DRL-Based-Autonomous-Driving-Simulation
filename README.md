# DRL-Autonomous-Driving-Simulation-ML-Agents
DRL(DQN) for AutonomousDriving Simulation using Unity:ML-Agents

## Environment
|OS|Language|IDE|
|:---:|:---:|:---:|
|Windows|C#, Python|Visual Studio, Visual Studio Code, Unity3D|  
<br> 
ML-Agents Version : 2.1.0 (Release 18)<br>  
Python Version : 3.7.9 <br> 
Unity Version : 2020.3.16f1 <br> 

## Main Reference
[RL Korea - RL Challenge with Unity](https://github.com/reinforcement-learning-kr/2021_RLKR_Drone_Delivery_Challenge_with_Unity)

## Code Overview
- `ML_test(ver9_fullsensor)` : Unity game environment (include `.exe` file)
- `DQN_Git` : DQN Algorithm and Trained Model

## Goal
Producing a model that determines behavior so that other vehicles, i.e., obstacles, can be avoided on the highway.
<br><br>
Actions that can be determined by the agent are determined(is 6 discrete Actions) and actions that can maximize future rewards with Deep reinforcement learning(Used DQN) are performed.

## Difficulties
- Learning DQN through papers but implementing it as a code is too hard.
- The surrounding environment and compensation must be set in detail, but there is no standard for this.
- If the environment or eward setting is strange, the test performance is not good even if the Reinforcement-learning is performed well.
