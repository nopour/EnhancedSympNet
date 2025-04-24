EnhancedSympNet-HenonHeiles

This repository contains a Python implementation of the EnhancedSympNet framework for simulating the Hénon-Heiles system, a chaotic Hamiltonian system. The code is based on the paper "EnhancedSympNet: A Neural-Symplectic Framework for Mechanical System Simulation" (2025). It combines symplectic integration (Velocity Verlet) with a neural network to learn correction dynamics, achieving improved accuracy and energy conservation compared to traditional methods.
EnhancedSympNet-HenonHeiles
===========================

Overview
--------
EnhancedSympNet-HenonHeiles is a Python-based implementation of the EnhancedSympNet framework for simulating the chaotic Henon-Heiles system, a nonlinear Hamiltonian system. Based on the paper "EnhancedSympNet: A Neural-Symplectic Framework for Mechanical System Simulation" by Reza Nopour and Afshin Taghvaeipour (2025), this repository contains clean and modular code for solving ordinary differential equations (ODEs) in mechanical systems using neural-symplectic integration. It combines symplectic Velocity Verlet with a neural correction network to achieve high accuracy and energy conservation.

Features
--------
- Modern implementation using Python and PyTorch
- Symplectic integration with adaptive time-stepping
- Physics-Informed Neural Networks (PINNs) and Hamiltonian Neural Networks (HNNs)
- Comprehensive visualizations: trajectories, phase space, Poincare sections, and more
- Ready to extend to other Hamiltonian systems (e.g., double pendulum, slider-crank)


Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~
- Python 3.9+
- Recommended: Create a virtual environment
```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

Usage
~~~~~
Run the main script:
```
python src/EnhacedSympNet_Hénon-Heiles.py
```

Example Output
--------------
Example terminal output:
```
Using device: cuda
Using dtype: torch.float32
Running simulation for: Mixed_E0p125 (Target E≈0.1250)
Initial State: q1=0.0000, p1=0.4906, q2=0.1000, p2=0.0000
Calculated Initial Energy H0 = 0.125000
Generating benchmark solution (3000000 pts)...
Benchmark energy drift: 1.296e-12 (Tol: 1.0e-08)

--- Train EnhancedSympNet(tanh) ---


Results saved to output/HH_Mixed_E0p125_results.xlsx
```

License
-------
This project is licensed under the MIT License. See the LICENSE file.

Acknowledgements
----------------
- Libraries: PyTorch (https://pytorch.org/), SciPy (https://scipy.org/), Matplotlib (https://matplotlib.org/)
- Special thanks to the open-source community


# Contact
Send any queries to Reza Nopour (rezanopour@gmail.com).


