EnhancedSympNet-HenonHeiles
This repository contains a Python implementation of the EnhancedSympNet framework for simulating the Hénon-Heiles system, a chaotic Hamiltonian system. The code is based on the paper "EnhancedSympNet: A Neural-Symplectic Framework for Mechanical System Simulation" by Reza Nopour and Afshin Taghvaeipour (2025). It combines symplectic integration (Velocity Verlet) with a neural network to learn correction dynamics, achieving improved accuracy and energy conservation compared to traditional methods.
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

Clone the repository:git clone https://github.com/<your-username>/EnhancedSympNet-HenonHeiles.git
cd EnhancedSympNet-HenonHeiles


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt

The requirements.txt file is included in the repository. Alternatively, install manually:pip install torch torchdiffeq numpy scipy matplotlib seaborn pandas openpyxl tqdm



Usage

Ensure dependencies are installed.
Run the main script to simulate the Hénon-Heiles system:python henon_heiles_simulation.py


The script will:
Generate a high-precision benchmark solution using SciPy.
Train EnhancedSympNet, Simple PINN, and Simple HNN models (100 epochs for testing).
Compare against Euler integration.
Produce plots (saved as PNG files) and export results to an Excel file (HH_Mixed_E0p125_*.xlsx).


Outputs are saved in the current directory with filenames prefixed by HH_Mixed_E0p125_T300_*.png.

Configuration
Modify the following parameters in henon_heiles_simulation.py to customize the simulation:

energy_level: Target initial energy (default: ( \frac{1}{8} )).
t_span_end: Simulation duration (default: 300 seconds).
num_points_train: Training time points (default: 6000).
num_points_eval: Evaluation time points (default: 600000).
epochs: Training epochs (default: 100 for testing).
batch_size: Training batch size (default: 32).
activations_to_test: Neural network activations (default: ['tanh']).

Example: To increase epochs to 1000, edit the script:
epochs = 1000

Directory Structure
EnhancedSympNet-HenonHeiles/
├── henon_heiles_simulation.py  # Main simulation script
├── requirements.txt            # Dependency list
├── README.md                   # This file
├── output/                     # (Generated) Plots and Excel files

Results and Visualizations
The script generates the following outputs:

Plots:
Trajectories (( q_1 ), ( q_2 ) vs. time).
Phase space (( q_1, p_1 ), ( q_2, p_2 )).
Normalized energy drift (( H/H_0 )).
Poincaré sections (( q_1 = 0, p_1 > 0 )).
RMSE vs. time (log scale).
Training loss history (total and component-wise).
3D phase space (( q_1, q_2, p_1 ), ( q_1, q_2, p_2 )).


Excel File: Time series data (( t, q_1, p_1, q_2, p_2, H_{\text{norm}}, \text{RMSE} )) for each method.

Sample plots are saved as HH_Mixed_E0p125_*.png. To view, check the output/ folder after running the script.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please report issues or suggestions via GitHub Issues.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

Based on the paper: Nopour, R., & Taghvaeipour, A. (2025). EnhancedSympNet: A Neural-Symplectic Framework for Mechanical System Simulation.
Thanks to the open-source community for libraries like PyTorch, SciPy, and Matplotlib.

Contact
For questions or support, please open an issue on GitHub or contact your-email@example.com.
