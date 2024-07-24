# REIL-Hex

This project was completed as a group effort by Nicolas Flake, Georg Johannes Talasz, and Marco Ross for the "Reinforcement Learning" module of our Master's studies at FH Technikum Wien. It involved training an agent to play the game of Hex and culminated in a competition where our agent won against all other participants of the course.

For a detailed explanation of our approaches and findings please read the the term paper.

## Overview

This repository contains the code and resources for training an agent to play the game of Hex using various reinforcement learning (RL) techniques. Our project focuses on evaluating different RL algorithms, training strategies, and neural network architectures to determine the most effective approach. The primary algorithms tested include Proximal Policy Optimization (PPO), AlphaZero, Soft Actor-Critic (SAC), and Deep Q-learning (DeepQ). The experiments were conducted on Hex board sizes of 3×3, 5×5, and 7×7.

## Key Findings

Our research demonstrated that Deep Q-learning, when combined with convolutional neural networks, achieved the highest overall performance. This approach not only converged quickly but also generalized well across different board sizes. AlphaZero showed potential but required significantly more resources and time for hyper-parameter tuning. Ultimately, the convolutional models trained with Deep Q-learning outperformed other architectures, making them the most effective solution for training an agent to play Hex.
