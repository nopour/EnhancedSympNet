EnhancedSympNet-HenonHeiles

This repository contains a Python implementation of the EnhancedSympNet framework for simulating the Hénon-Heiles system, a chaotic Hamiltonian system. The code is based on the paper "EnhancedSympNet: A Neural-Symplectic Framework for Mechanical System Simulation" (2025). It combines symplectic integration (Velocity Verlet) with a neural network to learn correction dynamics, achieving improved accuracy and energy conservation compared to traditional methods.

Features

Simulates the Hénon-Heiles system with initial energy ( E = \frac{1}{8} ) in the chaotic regime.
Implements EnhancedSympNet with adaptive time-stepping and a physics-informed loss function.
Compares against baselines: Euler, Simple PINN, and Simple HNN.
Generates visualizations: trajectories, phase space, Poincaré sections, energy drift, RMSE, and training loss.
Exports results to Excel for further analysis.
Supports CPU and GPU (CUDA) execution.

Prerequisites

Python: Version 3.8 or higher.
Dependencies:
torch (PyTorch for neural networks)
torchdiffeq (for ODE integration in PINN/HNN)
numpy (numerical computations)
scipy (high-precision benchmark solutions)
matplotlib (plotting)
seaborn (enhanced visualizations)
pandas and openpyxl (Excel export)
tqdm (progress bars)


Optional: CUDA-enabled GPU and compatible PyTorch version for faster training.

Installation

