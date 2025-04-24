# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plots
import seaborn as sns # For heatmap and palettes
from tqdm import tqdm
import os
import time
import warnings
import copy # For deep copying dictionaries
import re # For sanitizing sheet names

# --- Package Installation Helpers ---
def install_package(package):
    """Installs a package using pip."""
    print(f"Attempting to install {package}...")
    try:
        import subprocess
        import sys
        # Use check_call to raise an error if installation fails
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"{package} installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package} using pip: {e}")
        print(f"Please install {package} manually (e.g., 'pip install {package}') and restart the script.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during installation of {package}: {e}")
        return False

# Install torchdiffeq if not already installed
try:
    from torchdiffeq import odeint
except ImportError:
    if install_package("torchdiffeq"):
        from torchdiffeq import odeint
    else:
        print("Exiting script because torchdiffeq installation failed.")
        exit() # Exit if installation fails

# Install pandas and openpyxl for Excel export
try:
    import pandas as pd
except ImportError:
    if install_package("pandas"):
        import pandas as pd
    else:
        print("Exiting script because pandas installation failed.")
        exit()
try:
    import openpyxl
except ImportError:
    if install_package("openpyxl"):
        import openpyxl # Not strictly needed for basic ExcelWriter usage with pandas >= 1.2.0
    else:
        # Don't exit, just warn, as basic writing might work or user might use other formats
        print("Warning: openpyxl installation failed. Excel export might require manual installation ('pip install openpyxl').")


# --- Basic Setup ---
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32 # Use float32 for training efficiency

print(f"Using device: {DEVICE}")
print(f"Using dtype: {DTYPE}")

# --- Hénon-Heiles Specific Functions ---

def henon_heiles_dynamics(t, state):
    """ The exact Hénon-Heiles ODE equations. Handles tensors. """
    if isinstance(state, np.ndarray): state = torch.from_numpy(state).to(dtype=DTYPE, device=DEVICE)
    q1, p1, q2, p2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    dq1_dt = p1; dq2_dt = p2; dp1_dt = -q1 * (1 + 2 * q2); dp2_dt = -(q2 + q1**2 - q2**2)
    stack_dim = -1 if state.ndim > 1 else 0
    return torch.stack([dq1_dt, dp1_dt, dq2_dt, dp2_dt], dim=stack_dim)

def henon_heiles_hamiltonian(state):
    """ Calculates the Hénon-Heiles Hamiltonian. Handles different input shapes. """
    calc_dtype = state.dtype; dev = state.device
    # Determine dimensions and extract components
    if state.ndim >= 2: q1, p1, q2, p2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    elif state.ndim == 1: q1, p1, q2, p2 = state[0], state[1], state[2], state[3]
    else: raise ValueError(f"Unsupported state dimension: {state.ndim}")
    # Calculate terms
    p1sq=torch.pow(p1,2); p2sq=torch.pow(p2,2); q1sq=torch.pow(q1,2); q2sq=torch.pow(q2,2); q2cub=torch.pow(q2,3)
    half=torch.tensor(0.5, dtype=calc_dtype, device=dev); third=torch.tensor(1.0/3.0, dtype=calc_dtype, device=dev)
    # Calculate KE and PE
    ke=half*(p1sq+p2sq); pe=half*(q1sq+q2sq) + q1sq*q2 - third*q2cub; H = ke+pe
    return H.squeeze() if state.ndim == 1 else H

def exact_henon_heiles_solution(initial_state_np, t_span_end, num_points_benchmark=1000000, energy_tol=1e-8):
    """ Generates high-precision reference solution ('exact benchmark') using Scipy. """
    print(f"Generating benchmark solution ({num_points_benchmark} pts)...")
    t_eval_benchmark = np.linspace(0, t_span_end, num_points_benchmark)
    # Scipy ODE function wrapper requires float64
    def ode_scipy(t, state): return henon_heiles_dynamics(t, torch.from_numpy(state).to(torch.float64)).numpy()
    # Solve with high precision
    sol = solve_ivp(ode_scipy, [0, t_span_end], initial_state_np.astype(np.float64), t_eval=t_eval_benchmark, method='DOP853', rtol=1e-12, atol=1e-14)
    # Handle solver failure
    if not sol.success: print(f"Warning: Scipy failed! {sol.message}"); sol.y = np.full((4, num_points_benchmark), np.nan)
    # Process results
    states_np = sol.y.T # Shape (num_points_benchmark, 4)
    H = henon_heiles_hamiltonian(torch.tensor(states_np, dtype=torch.float64)).numpy()
    # Check energy drift
    initial_H = H[0] if len(H)>0 and np.isfinite(H[0]) else np.nan
    if not np.isnan(initial_H):
        drift=np.nanmax(np.abs(H-initial_H))
        print(f"Benchmark energy drift: {drift:.3e} (Tol: {energy_tol:.1e})")
        if drift > energy_tol: print("Warning: Benchmark drift exceeds tolerance.")
    else:
        print("Warning: Could not calculate benchmark energy drift (NaN initial energy).")
    return t_eval_benchmark, states_np, H

# --- Baseline Numerical Method ---
class EulerIntegrator:
    """ Simple Forward Euler integrator. """
    def predict(self, initial_state, t_eval):
        dt=(t_eval[1]-t_eval[0]).item(); n_steps=len(t_eval); state=initial_state.clone().to(DEVICE, DTYPE); states=[state]
        print(f"Running Euler ({n_steps} steps)...")
        with torch.no_grad(): # Ensure no gradients computed
            for i in tqdm(range(n_steps-1), desc="Euler", leave=False):
                # Ensure tensor is passed to dynamics
                derivs = henon_heiles_dynamics(t_eval[i].item(), state)
                state = state + dt * derivs
                # Check for NaN/Inf
                if not torch.isfinite(state).all():
                    print(f"\nEuler NaN detected at step {i+1}. Filling remaining steps.")
                    # Fill remaining with NaN
                    nan_state = torch.full_like(state, float('nan'))
                    states.extend([nan_state.clone() for _ in range(n_steps - (i + 1))])
                    break # Stop integration
                states.append(state.clone()) # Clone state at each step
        print("Euler done.")
        # Pad with NaNs if loop didn't complete
        if len(states) < n_steps:
            nan_state = torch.full_like(initial_state, float('nan'))
            states.extend([nan_state.clone() for _ in range(n_steps - len(states))])

        return torch.stack(states, dim=0) # Shape (time, 4)

# --- Learned Model Architectures ---

# 1. Correction Network for EnhancedSympNet
class CorrectionODEFunc(nn.Module):
    def __init__(self, h_dim=256, act='tanh'):
        super().__init__()
        act_fn={'tanh':nn.Tanh,'relu':nn.ReLU,'sigmoid':nn.Sigmoid,'gelu':nn.GELU}[act]
        self.net=nn.Sequential(nn.Linear(4,h_dim),act_fn(), nn.Linear(h_dim,h_dim),act_fn(), nn.Linear(h_dim,h_dim),act_fn(), nn.Linear(h_dim,1))
        self.activation=act
        self._init_weights() 

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                gain=nn.init.calculate_gain(self.activation if self.activation in ['tanh','relu','gelu'] else 'linear')
                nn.init.xavier_uniform_(m.weight, gain=gain); nn.init.zeros_(m.bias)

    def forward(self, t, state):
        with torch.enable_grad():
            state_g = state.detach().requires_grad_(True)
            H_c = self.net(state_g)
            grad_H = torch.autograd.grad(H_c.sum(), state_g, create_graph=True)[0]
        dq1=grad_H[:,1]; dp1=-grad_H[:,0]; dq2=grad_H[:,3]; dp2=-grad_H[:,2]
        correction_dynamics = torch.stack([dq1,dp1,dq2,dp2],dim=1)
        return correction_dynamics

# 2. Velocity Verlet Step
def symplectic_velocity_verlet_step(state, dt):
    """ Performs one step of the Velocity Verlet algorithm. State shape (batch, 4). """
    q1,p1,q2,p2=state[:,0],state[:,1],state[:,2],state[:,3]
    
    derivs=henon_heiles_dynamics(0.,state.to(DEVICE, DTYPE)); F1=derivs[:,1];F2=derivs[:,3] 
    p1h=p1+0.5*dt*F1; p2h=p2+0.5*dt*F2 
    q1n=q1+dt*p1h; q2n=q2+dt*p2h 
    
    state_n = torch.stack([q1n,p1h,q2n,p2h],dim=1).to(DEVICE, DTYPE)
    derivs_n=henon_heiles_dynamics(0.,state_n)
    F1n=derivs_n[:,1]; F2n=derivs_n[:,3]
    p1n=p1h+0.5*dt*F1n; p2n=p2h+0.5*dt*F2n 
    return torch.stack([q1n,p1n,q2n,p2n],dim=1)

# 3. EnhancedSympNet Model
class EnhancedSympNet(nn.Module):
    
    def __init__(self, h_dim=256, act='tanh', base_scale=0.005, beta=0.1):
        super().__init__()
        self.ode_func=CorrectionODEFunc(h_dim,act) 
        self.scale=nn.Parameter(torch.tensor(float(base_scale),dtype=DTYPE)) 
        self.beta=beta 
        self.model_type="EnhancedSympNet"; self.activation=act

    def forward(self, t_eval, state0):
        # state0 shape: (batch, 4)
        dt=(t_eval[1]-t_eval[0]).item(); n_steps=len(t_eval); states=[state0]; state=state0.clone()
        for i in range(n_steps-1):
            corr_dyn=self.ode_func(t_eval[i].item(), state)


            with torch.no_grad():
                correction_norm = torch.norm(corr_dyn, dim=1, keepdim=True)
                # Formula: factor = 1.0 - beta * ||correction||
                adaptive_factor = 1.0 - self.beta * correction_norm
                # Clamp the factor between 0.5 and 1.0 as described
                clamped_adaptive_factor = torch.clamp(adaptive_factor, min=0.5, max=1.0)
                # Calculate the adaptive time step
                adapt_dt = dt * clamped_adaptive_factor
            # --- MODIFICATION END: Adaptive Time-Stepping ---

            state_symp=symplectic_velocity_verlet_step(state, dt)
            
            state=state_symp + adapt_dt * (self.scale * corr_dyn)

            if not torch.isfinite(state).all():
                print(f"\n{self.model_type} ({self.activation}) NaN detected at step {i+1}. Filling remaining.")
                nan_state = torch.full_like(state, float('nan'))
                states.extend([nan_state.clone() for _ in range(n_steps - (i + 1))])
                break
            states.append(state.clone())
        if len(states) < n_steps:
            nan_state = torch.full_like(state0, float('nan'))
            states.extend([nan_state.clone() for _ in range(n_steps - len(states))])
        return torch.stack(states,dim=1) 

# 4. Simple PINN Model
class SimplePINN_ODEFunc(nn.Module): 
    def __init__(self, h_dim=256, act='tanh'):
        super().__init__(); act_fn={'tanh':nn.Tanh,'relu':nn.ReLU,'sigmoid':nn.Sigmoid,'gelu':nn.GELU}[act]
        self.net=nn.Sequential(nn.Linear(4+1,h_dim),act_fn(), nn.Linear(h_dim,h_dim),act_fn(), nn.Linear(h_dim,h_dim),act_fn(), nn.Linear(h_dim,4))
        self.activation=act; self.model_type="SimplePINN"; self._init_weights()
    def _init_weights(self): 
        for m in self.net.modules():
            if isinstance(m, nn.Linear): gain=nn.init.calculate_gain(self.activation if self.activation in ['tanh','relu','gelu'] else 'linear'); nn.init.xavier_uniform_(m.weight, gain=gain); nn.init.zeros_(m.bias)
    def forward(self, t, state): return self.net(torch.cat((state,torch.ones(state.shape[0],1,device=state.device)*t),dim=1))

# 5. Simple HNN Model
class SimpleHNN_ODEFunc(nn.Module): # Standard HNN: Learns H = NN(z)
    def __init__(self, h_dim=256, act='tanh'):
        super().__init__(); act_fn={'tanh':nn.Tanh,'relu':nn.ReLU,'sigmoid':nn.Sigmoid,'gelu':nn.GELU}[act]
        self.net=nn.Sequential(nn.Linear(4,h_dim),act_fn(), nn.Linear(h_dim,h_dim),act_fn(), nn.Linear(h_dim,h_dim),act_fn(), nn.Linear(h_dim,1))
        self.activation=act; self.model_type="SimpleHNN"; self._init_weights()
    def _init_weights(self): # Keep consistent initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear): gain=nn.init.calculate_gain(self.activation if self.activation in ['tanh','relu','gelu'] else 'linear'); nn.init.xavier_uniform_(m.weight, gain=gain); nn.init.zeros_(m.bias)
    def forward(self, t, state):
        with torch.enable_grad():
            state_g = state.clone().requires_grad_(True)
            H = self.net(state_g)
            grad_H = torch.autograd.grad(H.sum(), state_g, create_graph=True)[0]
        dq1=grad_H[:,1]; dp1=-grad_H[:,0]; dq2=grad_H[:,3]; dp2=-grad_H[:,2]
        dynamics = torch.stack([dq1,dp1,dq2,dp2],dim=1)
        return dynamics

# --- Loss Function
def physics_informed_loss_henon_heiles(model, states_pred, t_eval, initial_state_np, t_span_end):
    """ Calculates physics-informed loss. Handles NaNs. Consistent with 4.7.2 """
    # Input Validation
    if not torch.isfinite(states_pred).all():
         print("Loss NaN Input."); high_loss, nan_loss = torch.tensor(1e7,device=DEVICE,dtype=DTYPE), torch.tensor(float('nan'),device=DEVICE,dtype=DTYPE)
         return high_loss, nan_loss, nan_loss, nan_loss, nan_loss, nan_loss, nan_loss

    b, n_t, _ = states_pred.shape;
    # Handle case where n_t might be 1
    if n_t <= 1:
        print("Warning: Loss function received states_pred with insufficient time points."); return torch.tensor(1e7,device=DEVICE,dtype=DTYPE), *(6 * [torch.tensor(float('nan'),device=DEVICE,dtype=DTYPE)])

    dt = (t_eval[1]-t_eval[0]).item()
    q1, p1, q2, p2 = states_pred[...,0], states_pred[...,1], states_pred[...,2], states_pred[...,3]

    # Calculate initial energy robustly
    initial_state_tensor = torch.tensor(initial_state_np, dtype=torch.float64, device=DEVICE)
    E0 = henon_heiles_hamiltonian(initial_state_tensor).item()
    norm_f = max(abs(E0), 1e-6) # Normalization factor

    # Time-Dependent Loss Weighting
    # Ensure time weights are broadcastable (shape [n_t-1] or [n_t])
    # Use coefficient 0.5 to match Section 4.7.2: w(t) = 1.0 + 0.5 * (t / T)
    time_weight_coeff = 0.5
    t_w_vec = (1.0 + time_weight_coeff * (t_eval[:-1] / t_span_end)).to(DEVICE, DTYPE) # Shape [n_t-1]
    t_w_full_vec = (1.0 + time_weight_coeff * (t_eval / t_span_end)).to(DEVICE, DTYPE)  # Shape [n_t]
    # Time-Dependent Loss Weighting

    # 1. ODE Residual Loss 
    dq1dt_p=(q1[:,1:]-q1[:,:-1])/dt; dp1dt_p=(p1[:,1:]-p1[:,:-1])/dt; dq2dt_p=(q2[:,1:]-q2[:,:-1])/dt; dp2dt_p=(p2[:,1:]-p2[:,:-1])/dt
    states_mid = 0.5 * (states_pred[:, :-1, :] + states_pred[:, 1:, :])
    valid_mid = torch.isfinite(states_mid)
    derivs_t = torch.full_like(states_mid, float('nan'))
    if valid_mid.all():
        derivs_t = henon_heiles_dynamics(0., states_mid.reshape(-1, 4)).reshape(b, n_t - 1, 4)
    else: print("Warning: NaNs detected in midpoint states for ODE loss calculation.")
    dq1dt_t, dp1dt_t, dq2dt_t, dp2dt_t = derivs_t[...,0], derivs_t[...,1], derivs_t[...,2], derivs_t[...,3]
    ode_res_sq = (dq1dt_p-dq1dt_t)**2+(dp1dt_p-dp1dt_t)**2+(dq2dt_p-dq2dt_t)**2+(dp2dt_p-dp2dt_t)**2
    valid_ode_res = torch.isfinite(ode_res_sq)
    # 
    weighted_ode_res_sq = t_w_vec * ode_res_sq # 
    ode_loss = torch.mean(weighted_ode_res_sq[valid_ode_res]) if valid_ode_res.any() else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    # 2. Initial Condition Loss
    state0_true_b = torch.tensor(initial_state_np,dtype=DTYPE,device=DEVICE).unsqueeze(0).repeat(b,1)
    ic_loss = F.mse_loss(states_pred[:,0,:], state0_true_b)

    # 3. Energy Conservation Loss 
    E_pred = henon_heiles_hamiltonian(states_pred.to(torch.float64)).to(DTYPE)
    energy_res_sq = ((E_pred - E0) / norm_f)**2
    valid_energy_res = torch.isfinite(energy_res_sq)
    # Apply weighting 
    weighted_energy_res_sq = t_w_full_vec * energy_res_sq #
    energy_loss = torch.mean(weighted_energy_res_sq[valid_energy_res]) if valid_energy_res.any() else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    # 4. Energy Gradient Loss (dE/dt should be zero)
    dEdt = (E_pred[:,1:]-E_pred[:,:-1])/dt
    energy_grad_res_sq = (dEdt / norm_f)**2
    valid_energy_grad = torch.isfinite(energy_grad_res_sq)
    
    weighted_energy_grad_res_sq = t_w_vec * energy_grad_res_sq # Broadcasts t_w_vec along batch dim
    energy_grad_loss = torch.mean(weighted_energy_grad_res_sq[valid_energy_grad]) if valid_energy_grad.any() else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    # 5. Symplectic Loss
    symp_loss = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    if isinstance(model, EnhancedSympNet) and model.training and (torch.rand(1).item()<0.1):
        subset=min(20, b*(n_t-1));
        if subset > 0 and valid_mid.all(): 
            indices=torch.randperm(b*(n_t-1),device=DEVICE)[:subset]
            state_sub = states_mid.reshape(-1, 4)[indices]
            with torch.enable_grad(): state_sub_g = state_sub.detach().requires_grad_(True)
            try:
                div=[]
                for k in range(subset):
                    def single_sample_corr_dyn(x_single): return model.ode_func(0., x_single.unsqueeze(0)).squeeze(0)
                    jac_k=torch.autograd.functional.jacobian(single_sample_corr_dyn, state_sub_g[k], create_graph=False)
                    div.append(torch.trace(jac_k))
                if div: symp_loss=F.mse_loss(torch.stack(div),torch.zeros(subset,device=DEVICE))
            except Exception as e: print(f"Symplectic Jacobian calculation error: {e}")

    # 6. L2 Regularization
    l2_loss = torch.tensor(0.0,device=DEVICE,dtype=DTYPE)
    if isinstance(model, nn.Module): l2_loss=sum(p.pow(2.).sum() for p in model.parameters())

    # Total Loss
    lw={'ode':10.,'ic':100.,'energy':25.,'energy_grad':5.,'symplectic':0.1,'l2':1e-7}
    total_loss=(lw['ode'] * ode_loss + lw['ic'] * ic_loss + lw['energy'] * energy_loss +
                lw['energy_grad'] * energy_grad_loss + lw['symplectic'] * symp_loss + lw['l2'] * l2_loss)

    # Final check for safety
    if not torch.isfinite(total_loss):
        print(f"Loss NaN Final (ODE:{ode_loss.item():.2e}, IC:{ic_loss.item():.2e}, E:{energy_loss.item():.2e}, EGrad:{energy_grad_loss.item():.2e}, Symp:{symp_loss.item():.2e}, L2:{l2_loss.item():.2e}).");
        total_loss=torch.tensor(1e8,device=DEVICE,dtype=DTYPE)

    return total_loss, ode_loss, ic_loss, energy_loss, energy_grad_loss, symp_loss, l2_loss


#Training Loop 
def train_model(model, initial_state_np, t_span_end, num_points_train=2000, epochs=150, batch_size=64, lr=1e-4, patience=50):
    model_name=model.__class__.__name__; act=getattr(model,'activation','N/A'); print(f"\n--- Train {model_name}({act}) ---")
    start_t=time.time(); optimizer=torch.optim.Adam(model.parameters(),lr=lr); scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,epochs//3),gamma=0.5)
    t_train=torch.linspace(0,t_span_end,num_points_train,dtype=DTYPE,device=DEVICE);
    state0_batch=torch.tensor(initial_state_np,dtype=DTYPE,device=DEVICE).unsqueeze(0).repeat(batch_size,1)
    model.to(DEVICE); hist={k:[] for k in ['total','ode','ic','energy','energy_grad','symplectic','l2','epoch','lr']}
    best_loss, no_improve = float('inf'), 0
    use_odeint=isinstance(model,(SimplePINN_ODEFunc, SimpleHNN_ODEFunc))

    for epoch in tqdm(range(epochs), desc=f"Train {model_name}({act})"):
        model.train(); optimizer.zero_grad()
        try: # Integration Step
            if use_odeint:
                pred_states = odeint(model, state0_batch, t_train, method='dopri5', rtol=1e-6, atol=1e-7).permute(1,0,2)
            else: # EnhancedSympNet uses custom forward
                pred_states = model(t_train, state0_batch)
            if not torch.isfinite(pred_states).all(): raise ValueError("NaN detected in predicted states during training.")
        except Exception as e: # Catch errors during forward pass or NaN check
            print(f"\nForward Pass Error E{epoch+1}: {e}")
            loss_val = np.nan; hist['epoch'].append(epoch+1); hist['lr'].append(optimizer.param_groups[0]['lr']); [hist[k].append(np.nan) for k in hist if k not in ['epoch','lr']]
            no_improve += 1
            if no_improve >= patience: print(f"\nEarly Stop E{epoch+1} due to consecutive errors/no improvement."); break
            continue 

        # Loss Calculation
        loss_tuple = physics_informed_loss_henon_heiles(model, pred_states, t_train, initial_state_np, t_span_end)
        total_loss=loss_tuple[0]; loss_val=total_loss.item()

        # Backpropagation
        if not torch.isfinite(total_loss):
            print(f"E{epoch+1}: NaN Loss ({loss_val:.3e}). Skip Bwd.")
            loss_val=np.nan; no_improve +=1
        else:
            try:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()) 
                if not grads_finite:
                    print(f"\nE{epoch+1}: NaN/Inf Gradients Detected. Skip Opt.")
                    optimizer.zero_grad(); no_improve +=1
                else:
                    optimizer.step()
                    if loss_val < best_loss * 0.999: best_loss = loss_val; no_improve = 0
                    else: no_improve += 1
            except RuntimeError as e:
                 print(f"\nRuntimeError during backward E{epoch+1}: {e}")
                 loss_val = np.nan; no_improve +=1

        
        scheduler.step()
        if no_improve >= patience:
            print(f"\nEarly Stop E{epoch+1} (Patience {patience} exceeded). Best Loss: {best_loss:.3e}")
            break

        
        hist['epoch'].append(epoch+1); hist['lr'].append(optimizer.param_groups[0]['lr'])
        for i,k in enumerate(['total','ode','ic','energy','energy_grad','symplectic','l2']):
            item_val = loss_tuple[i].item() if torch.is_tensor(loss_tuple[i]) and torch.isfinite(loss_tuple[i]).all() else np.nan
            hist[k].append(item_val if k!='total' else loss_val) # Log the loss_val used for checks

        if (epoch+1)%50==0 and not np.isnan(loss_val): tqdm.write(f"E{epoch+1}, L:{loss_val:.3e}, LR:{optimizer.param_groups[0]['lr']:.2e}, BestL:{best_loss:.3e}, Patience:{no_improve}/{patience}")

    print(f"Train {model_name}({act}) done ({time.time()-start_t:.1f}s). Final Best Loss: {best_loss:.3e}")
    if DEVICE=='cuda': torch.cuda.empty_cache()
    return hist


# --- Prediction and Analysis Functions
def compute_poincare(t, states, cross_idx=0, cross_val=0.0, pos_dir=True):
    valid=np.isfinite(states).all(axis=1); t,states=t[valid],states[valid]; n=len(t)
    if n<2: return np.array([]),np.array([])
    pts_q,pts_p=[],[]; crs=states[:,cross_idx]; mom=states[:,cross_idx+1] if cross_idx%2==0 else None
    pq_idx=2 if cross_idx in [0,1] else 0; pp_idx=pq_idx+1; pq,pp=states[:,pq_idx],states[:,pp_idx]
    for i in range(n-1):
        ci,cip1=crs[i]-cross_val,crs[i+1]-cross_val
        if ci*cip1 < 0:
            dir_ok = True
            if mom is not None:
                dc = cip1 - ci
                if abs(dc) > 1e-15: w = -ci / dc; interp_mom = mom[i] + w * (mom[i+1] - mom[i]); m = interp_mom
                else: m = 0.5 * (mom[i] + mom[i+1])
                if (pos_dir and m < 0) or (not pos_dir and m > 0): dir_ok = False
            if dir_ok:
                dc = cip1 - ci
                if abs(dc) > 1e-15:
                    w = np.clip(-ci/dc, 0., 1.)
                    iq = pq[i] + w * (pq[i+1]-pq[i]); ip = pp[i] + w * (pp[i+1]-pp[i])
                    if np.isfinite(iq) and np.isfinite(ip): pts_q.append(iq); pts_p.append(ip)
    return np.array(pts_q), np.array(pts_p)

def calculate_rmse_over_time(pred_states, true_states, true_t, eval_t):
    pred_np = pred_states.cpu().numpy() if isinstance(pred_states,torch.Tensor) else np.array(pred_states)
    true_np = true_states.cpu().numpy() if isinstance(true_states,torch.Tensor) else np.array(true_states)
    true_t_np = true_t.cpu().numpy() if isinstance(true_t,torch.Tensor) else np.array(true_t)
    eval_t_np = eval_t.cpu().numpy() if isinstance(eval_t,torch.Tensor) else np.array(eval_t)
    if not np.isfinite(true_t_np).all() or not np.isfinite(true_np).all(): print("Warning: Benchmark data contains NaNs/Infs. Cannot compute valid RMSE."); return np.full(len(eval_t_np), np.nan)
    if len(true_t_np) < 2: print("Warning: Benchmark data has insufficient points for interpolation."); return np.full(len(eval_t_np), np.nan)
    if eval_t_np.min() < true_t_np.min() or eval_t_np.max() > true_t_np.max(): print(f"Warning: Eval times [{eval_t_np.min():.2f}, {eval_t_np.max():.2f}] outside benchmark times [{true_t_np.min():.2f}, {true_t_np.max():.2f}]. RMSE inaccurate.")
    try:
        interp_funcs = [interp1d(true_t_np, true_np[:, i], kind='linear', bounds_error=False, fill_value=np.nan) for i in range(true_np.shape[1])]
        true_states_interp = np.stack([func(eval_t_np) for func in interp_funcs], axis=-1)
    except ValueError as e: print(f"Error during interpolation for RMSE: {e}. Returning NaNs."); return np.full(len(eval_t_np), np.nan)
    if pred_np.shape != true_states_interp.shape: print(f"Warning: RMSE shape mismatch - Pred {pred_np.shape}, True_interp {true_states_interp.shape}. Returning NaNs."); return np.full(len(eval_t_np), np.nan)
    valid_pred = np.isfinite(pred_np).all(axis=1); valid_true = np.isfinite(true_states_interp).all(axis=1); valid = valid_pred & valid_true
    rmse=np.full(len(eval_t_np),np.nan)
    if np.any(valid): diff_sq = (pred_np[valid] - true_states_interp[valid])**2; mean_sq_err = np.mean(diff_sq, axis=1); rmse[valid] = np.sqrt(mean_sq_err)
    return rmse

def generate_predictions(models_dict, exact_data_highres, initial_state_np, t_span_end, num_points_eval):
    print("\nGenerating predictions for all methods...")
    t_eval = torch.linspace(0, t_span_end, num_points_eval, dtype=DTYPE, device=DEVICE)
    t_eval_np = t_eval.cpu().numpy()
    state0 = torch.tensor(initial_state_np, dtype=DTYPE, device=DEVICE)
    state0_batch = state0.unsqueeze(0)
    predictions = {}

    # --- Add Exact Benchmark Data (Interpolated) ---
    print("  Processing Exact Benchmark Data...")
    t_benchmark = exact_data_highres['t']; states_benchmark = exact_data_highres['states']; H_benchmark = exact_data_highres['H']
    if not np.isfinite(t_benchmark).all() or not np.isfinite(states_benchmark).all() or not np.isfinite(H_benchmark).all():
        print("Error: High-resolution benchmark data contains NaNs/Infs. Cannot proceed with predictions.")
        predictions['Exact'] = {'states': np.full((num_points_eval, 4), np.nan), 'H_norm': np.full(num_points_eval, np.nan), 't_eval': t_eval_np, 'rmse_vs_time': np.full(num_points_eval, np.nan), 'history': {}}
    else:
        initial_energy = H_benchmark[0]
        try:
            interp_funcs_states = [interp1d(t_benchmark, states_benchmark[:, i], kind='linear', bounds_error=False, fill_value=np.nan) for i in range(states_benchmark.shape[1])]
            exact_states_eval = np.stack([func(t_eval_np) for func in interp_funcs_states], axis=-1)
            interp_func_H = interp1d(t_benchmark, H_benchmark, kind='linear', bounds_error=False, fill_value=np.nan)
            exact_H_eval = interp_func_H(t_eval_np)
            exact_H_norm_eval = exact_H_eval / max(abs(initial_energy), 1e-9)
            predictions['Exact'] = {'states': exact_states_eval, 'H_norm': exact_H_norm_eval, 't_eval': t_eval_np, 'rmse_vs_time': np.zeros(num_points_eval), 'history': {}}
            print("    Exact data interpolated successfully.")
        except Exception as e:
            print(f"    ERROR interpolating exact data: {e}")
            predictions['Exact'] = {'states': np.full((num_points_eval, 4), np.nan), 'H_norm': np.full(num_points_eval, np.nan), 't_eval': t_eval_np, 'rmse_vs_time': np.full(num_points_eval, np.nan), 'history': {}}

    # --- Generate Predictions for Other Models ---
    models_to_process = copy.deepcopy(models_dict)
    initial_energy_ref = exact_data_highres['H'][0] if 'H' in exact_data_highres and len(exact_data_highres['H']) > 0 and np.isfinite(exact_data_highres['H'][0]) else np.nan

    for label, data in models_to_process.items():
        if label.lower() == 'exact': continue
        model = data.get('model'); start_t = time.time(); states_pred_np = None; print(f"  Predicting: {label}")
        try:
            if isinstance(model, EulerIntegrator):
                states_pred_np = model.predict(state0.to(DEVICE), t_eval).cpu().numpy()
            elif isinstance(model, nn.Module):
                model.eval().to(DEVICE)
                if isinstance(model, EnhancedSympNet):
                     with torch.no_grad():
                         pred_t = model(t_eval, state0_batch); states_pred_np = pred_t.squeeze(0).cpu().numpy()
                else: # PINN or HNN using odeint
                    with torch.no_grad(): 
                        pred_t = odeint(model, state0_batch, t_eval, method='dopri5', rtol=1e-7, atol=1e-8); states_pred_np = pred_t.squeeze(1).cpu().numpy()
            else: raise TypeError(f"Unsupported model type for label {label}")

            if not np.isfinite(states_pred_np).all(): print(f"    WARNING: NaNs detected in prediction output for {label}.")
            if states_pred_np.shape != (num_points_eval, 4):
                print(f"    Warning: Prediction shape mismatch for {label}. Expected ({num_points_eval}, 4), got {states_pred_np.shape}. Padding/reshaping...");
                padded_states = np.full((num_points_eval, 4), np.nan)
                valid_len = min(len(states_pred_np), num_points_eval);
                if valid_len > 0 and states_pred_np.ndim == 2 and states_pred_np.shape[1]==4: padded_states[:valid_len, :] = states_pred_np[:valid_len, :]
                states_pred_np = padded_states
        except Exception as e: print(f"    ERROR predicting {label}: {e}"); states_pred_np = np.full((num_points_eval, 4), np.nan)

        # Post-process
        valid = np.isfinite(states_pred_np).all(axis=1); H_norm = np.full(num_points_eval, np.nan)
        if np.any(valid) and not np.isnan(initial_energy_ref): H_pred=henon_heiles_hamiltonian(torch.tensor(states_pred_np[valid],dtype=torch.float64)).numpy(); H_norm[valid]=H_pred/max(abs(initial_energy_ref),1e-9)
        elif np.isnan(initial_energy_ref): print(f"    Warning: Cannot calculate normalized energy for {label} due to invalid benchmark initial energy.")
        rmse = calculate_rmse_over_time(states_pred_np, states_benchmark, t_benchmark, t_eval_np)
        predictions[label] = {'states':states_pred_np, 'H_norm':H_norm, 't_eval':t_eval_np, 'rmse_vs_time':rmse, 'history':data.get('history',{})}
        max_rmse_val = np.nanmax(rmse) if np.any(np.isfinite(rmse)) else np.nan
        print(f"    {label} Pred done ({time.time()-start_t:.1f}s). Max RMSE: {max_rmse_val:.2e}")
        if DEVICE=='cuda':
             if 'pred_t' in locals(): del pred_t
             if 'model' in locals() and isinstance(model, nn.Module): model.to('cpu')
             torch.cuda.empty_cache()
    return predictions


# --- Plotting Utilities ---
def _get_plot_styles(predictions_subset):
    """Helper to create consistent plot styles."""
    n_methods = len(predictions_subset)
    try: color_palette = sns.color_palette("colorblind", n_methods)
    except: color_palette = sns.color_palette("husl", n_methods)
    styles = {}; color_idx = 0
    for i, label in enumerate(predictions_subset.keys()):
        style = {'alpha': 0.85, 'lw': 1.5}
        if label == 'Exact': style.update({'color': 'black', 'ls': '-', 'lw': 0.8, 'alpha': 0.9, 'zorder': 50})
        elif 'Euler' in label: style.update({'color': color_palette[color_idx % len(color_palette)], 'ls': ':', 'lw': 1.2, 'alpha': 0.7, 'zorder': 10}); color_idx+=1 # Consume a color index
        elif 'Enhanced' in label:
            # 
            style.update({'color': 'firebrick', 'ls':'-', 'lw':1.6, 'zorder':40})
            # 
        elif 'PINN' in label: style.update({'color': color_palette[color_idx % len(color_palette)], 'ls':'--', 'lw':1.5, 'zorder':30}); color_idx+=1
        elif 'HNN' in label: style.update({'color': color_palette[color_idx % len(color_palette)], 'ls':'-.', 'lw':1.5, 'zorder':20}); color_idx+=1
        else: style.update({'color': color_palette[color_idx % len(color_palette)], 'ls':'-', 'lw':1.2, 'zorder':5 }); color_idx+=1
        styles[label] = style
    return styles

# --- Plotting Functions 
# Plotting Function 1: Comparison Set
def plot_comparison_set(predictions_subset, exact_data_highres, activation_label, base_filename, plot_config):
    print(f"\n--- Plots: Comparison Set (Activation: {activation_label}) ---")
    cfg_glob = plot_config.get('global', {}); dpi = cfg_glob.get('dpi', 300); default_skip_ratio = cfg_glob.get('plot_skip_ratio', 1000)
    phase_ms = cfg_glob.get('phase_marker_size', 1); poincare_ms = cfg_glob.get('poincare_marker_size', 4)
    n_methods = len(predictions_subset)
    if n_methods <= 1: print("   Skipping plot - only one method found."); return
    styles = _get_plot_styles(predictions_subset) # Use the helper function
    ref_data = predictions_subset.get('Exact', next(iter(predictions_subset.values()), None))
    num_points_plot = len(ref_data['t_eval']) if ref_data and 't_eval' in ref_data else 0
    if num_points_plot == 0: print("   Skipping plot - reference data has no time points."); return
    plot_skip = max(1, num_points_plot // default_skip_ratio)
    filename_prefix = f"{base_filename}_Compare_Act_{activation_label}"

    # Plot 1: Trajectories
    cfg_traj = plot_config.get('trajectory', {}); fig1=None
    try:
        fig1, axs1 = plt.subplots(2, 1, figsize=cfg_traj.get('figsize', (12, 8)), sharex=True); plotted_any = False
        for label, data in predictions_subset.items():
            s=styles.get(label, styles.get('Exact')); t=data.get('t_eval'); st=data.get('states')
            if t is not None and st is not None and st.ndim == 2 and st.shape[1] >= 3:
                valid=np.isfinite(st[:,0]) & np.isfinite(st[:,2])
                if np.any(valid): axs1[0].plot(t[::plot_skip][valid[::plot_skip]], st[::plot_skip,0][valid[::plot_skip]], label=f'{label} $q_1$', **s); axs1[1].plot(t[::plot_skip][valid[::plot_skip]], st[::plot_skip,2][valid[::plot_skip]], label=f'{label} $q_2$', **s); plotted_any = True
            else: print(f"    Warning: Invalid/missing trajectory data for {label}")
        if not plotted_any: print("    Skipping trajectory plot - no valid data to plot."); plt.close(fig1); return
        axs1[0].set_ylabel('$q_1$'); axs1[0].grid(True, ls=':'); axs1[0].legend(fontsize=cfg_traj.get('legend_fontsize', 8), loc='best')
        if cfg_traj.get('q1_ylim'): axs1[0].set_ylim(cfg_traj['q1_ylim'])
        axs1[1].set_ylabel('$q_2$'); axs1[1].grid(True, ls=':'); axs1[1].set_xlabel('Time'); axs1[1].legend(fontsize=cfg_traj.get('legend_fontsize', 8), loc='best')
        if cfg_traj.get('q2_ylim'): axs1[1].set_ylim(cfg_traj['q2_ylim'])
        if cfg_traj.get('time_xlim'): axs1[1].set_xlim(cfg_traj['time_xlim'])
        fig1.suptitle(f'Trajectories Comparison (Act: {activation_label})'); fig1.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(f'{filename_prefix}_traj.png', dpi=dpi)
    except Exception as e: print(f"Error plotting trajectories: {e}")
    finally: plt.close(fig1) if fig1 else None

    # Plot 2: Phase Space
    cfg_phase = plot_config.get('phase', {}); fig2=None
    try:
        fig2, axs2 = plt.subplots(1, 2, figsize=cfg_phase.get('figsize', (14, 6))); plotted_any = False
        for label, data in predictions_subset.items():
            s=styles.get(label, styles.get('Exact')); st=data.get('states')
            if st is not None and st.ndim == 2 and st.shape[1] == 4:
                valid = np.isfinite(st).all(axis=1)
                if np.any(valid): axs2[0].plot(st[::plot_skip,0][valid[::plot_skip]], st[::plot_skip,1][valid[::plot_skip]], '.', ms=phase_ms, label=label, color=s['color'], alpha=s['alpha'], zorder=s.get('zorder',1)); axs2[1].plot(st[::plot_skip,2][valid[::plot_skip]], st[::plot_skip,3][valid[::plot_skip]], '.', ms=phase_ms, label=label, color=s['color'], alpha=s['alpha'], zorder=s.get('zorder',1)); plotted_any = True
            else: print(f"    Warning: Invalid/missing phase data for {label}")
        if not plotted_any: print("    Skipping phase plot - no valid data to plot."); plt.close(fig2); return
        axs2[0].set_xlabel('$q_1$'); axs2[0].set_ylabel('$p_1$'); axs2[0].set_title('$(q_1, p_1)$'); axs2[0].grid(True, ls=':'); axs2[0].legend(fontsize=cfg_phase.get('legend_fontsize', 7), markerscale=cfg_phase.get('legend_markerscale', 5));
        if cfg_phase.get('q1p1_xlim'): axs2[0].set_xlim(cfg_phase['q1p1_xlim'])
        if cfg_phase.get('q1p1_ylim'): axs2[0].set_ylim(cfg_phase['q1p1_ylim'])
        if cfg_phase.get('equal_aspect', True): axs2[0].set_aspect('equal', adjustable='box')
        axs2[1].set_xlabel('$q_2$'); axs2[1].set_ylabel('$p_2$'); axs2[1].set_title('$(q_2, p_2)$'); axs2[1].grid(True, ls=':'); axs2[1].legend(fontsize=cfg_phase.get('legend_fontsize', 7), markerscale=cfg_phase.get('legend_markerscale', 5));
        if cfg_phase.get('q2p2_xlim'): axs2[1].set_xlim(cfg_phase['q2p2_xlim'])
        if cfg_phase.get('q2p2_ylim'): axs2[1].set_ylim(cfg_phase['q2p2_ylim'])
        if cfg_phase.get('equal_aspect', True): axs2[1].set_aspect('equal', adjustable='box')
        fig2.suptitle(f'Phase Space Comparison (Act: {activation_label})'); fig2.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(f'{filename_prefix}_phase.png', dpi=dpi)
    except Exception as e: print(f"Error plotting phase space: {e}")
    finally: plt.close(fig2) if fig2 else None

    # Plot 3: Energy
    cfg_energy = plot_config.get('energy', {}); fig3=None
    try:
        fig3, ax3 = plt.subplots(1, 1, figsize=cfg_energy.get('figsize', (10, 5))); min_E, max_E = 1.0, 1.0; has_valid_energy = False
        for label, data in predictions_subset.items():
            s=styles.get(label, styles.get('Exact')); h_norm=data.get('H_norm'); t=data.get('t_eval')
            if h_norm is not None and t is not None:
                valid=np.isfinite(h_norm);
                if np.any(valid): ax3.plot(t[valid], h_norm[valid], label=label, **s); min_E = min(min_E, np.nanmin(h_norm[valid])); max_E = max(max_E, np.nanmax(h_norm[valid])); has_valid_energy = True
            else: print(f"    Warning: Invalid/missing energy data for {label}")
        if not has_valid_energy: print("    Skipping energy plot - no valid data to plot."); plt.close(fig3); return
        ax3.set_xlabel('Time'); ax3.set_ylabel('Norm. Energy (H/H$_0$)'); ax3.set_title(f'Energy Comparison (Act: {activation_label})'); ax3.grid(True, ls=':'); ax3.legend(fontsize=cfg_energy.get('legend_fontsize', 'small'))
        force_ylim = cfg_energy.get('force_ylim');
        if force_ylim: ax3.set_ylim(force_ylim)
        else: rel_range = cfg_energy.get('ylim_rel_range', 1.3); energy_range = max(max_E - 1.0, 1.0 - min_E, 0.01); ax3.set_ylim(1.0 - energy_range * rel_range, 1.0 + energy_range * rel_range)
        if cfg_traj.get('time_xlim'): ax3.set_xlim(cfg_traj['time_xlim'])
        fig3.tight_layout(); plt.savefig(f'{filename_prefix}_energy.png', dpi=dpi)
    except Exception as e: print(f"Error plotting energy: {e}")
    finally: plt.close(fig3) if fig3 else None

    # Plot 4: Poincaré
    cfg_poincare = plot_config.get('poincare', {}); fig4=None
    try:
        fig4, ax4 = plt.subplots(1, 1, figsize=cfg_poincare.get('figsize', (8, 8))); print(f"  Poincaré ({activation_label})..."); plotted_any = False
        poincare_data = {}; t_hr, s_hr = exact_data_highres.get('t'), exact_data_highres.get('states'); qe_hr, pe_hr = ([], [])
        if t_hr is not None and s_hr is not None: qe_hr, pe_hr = compute_poincare(t_hr, s_hr); poincare_data['Exact_HighRes'] = (qe_hr, pe_hr)
        else: print("    Warning: Missing high-res exact data for Poincaré reference.")
        for lbl, d in predictions_subset.items():
            if lbl.lower() != 'exact': t_ev, s_ev = d.get('t_eval'), d.get('states');
            if t_ev is not None and s_ev is not None: poincare_data[lbl] = compute_poincare(t_ev, s_ev)
        plotted_exact = False
        if 'Exact_HighRes' in poincare_data:
            qe_hr, pe_hr = poincare_data['Exact_HighRes']
            if len(qe_hr) > 0:
                se = styles['Exact']; ax4.plot(qe_hr, pe_hr, '.', ms=max(1, poincare_ms // 2), label=f'Exact ({len(qe_hr)}pts)', color=se['color'], alpha=se['alpha']*0.7, zorder=se.get('zorder', 50)-1); plotted_exact = True; plotted_any = True
                qmn_e, qmx_e = np.min(qe_hr), np.max(qe_hr); pmn_e, pmx_e = np.min(pe_hr), np.max(pe_hr); qr_e = max(qmx_e - qmn_e, 1e-2); pr_e = max(pmx_e - pmn_e, 1e-2); default_xlim = (qmn_e - 0.1 * qr_e, qmx_e + 0.1 * qr_e); default_ylim = (pmn_e - 0.1 * pr_e, pmx_e + 0.1 * pr_e); ax4.set_xlim(cfg_poincare.get('xlim', default_xlim)); ax4.set_ylim(cfg_poincare.get('ylim', default_ylim))
        for label, data in poincare_data.items():
            if label != 'Exact_HighRes' and label in styles:
                q,p = data
                if len(q) > 0: s = styles[label]; ax4.plot(q, p, '.', ms=poincare_ms, label=f'{label} ({len(q)}pts)', color=s['color'], alpha=s['alpha'], zorder=s.get('zorder', 1)); plotted_any = True
                elif label != 'Exact_HighRes': print(f"    Warning: No Poincaré points for {label}")
        if not plotted_any: print("    Skipping Poincaré plot - no valid data to plot."); plt.close(fig4); return
        ax4.set_xlabel('$q_2$'); ax4.set_ylabel('$p_2$'); ax4.set_title(f'Poincaré ($q_1=0, p_1>0$, Act: {activation_label})'); ax4.legend(fontsize=cfg_poincare.get('legend_fontsize', 'small'), markerscale=cfg_poincare.get('legend_markerscale', 3)); ax4.grid(True, ls=':');
        if cfg_poincare.get('equal_aspect', True): ax4.set_aspect('equal', adjustable='box')
        fig4.tight_layout(); plt.savefig(f'{filename_prefix}_poincare.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Poincare: {e}")
    finally: plt.close(fig4) if fig4 else None

    # Plot 5: RMSE
    cfg_rmse = plot_config.get('rmse', {}); fig5=None
    try:
        fig5, ax5 = plt.subplots(1, 1, figsize=cfg_rmse.get('figsize', (10, 5))); has_rmse = False
        for label, data in predictions_subset.items():
            if label.lower() != 'exact':
                s=styles.get(label, styles.get('Exact')); rmse = data.get('rmse_vs_time'); t=data.get('t_eval')
                if rmse is not None and t is not None:
                    valid=np.isfinite(rmse)
                    if np.any(valid): ax5.plot(t[valid], rmse[valid], label=label, **s); has_rmse = True
                    else: print(f"    Warning: Invalid/missing RMSE data for {label}")
        if not has_rmse: print("    Skipping RMSE plot - no valid data to plot."); plt.close(fig5); return
        ax5.set_xlabel('Time'); ax5.set_ylabel('RMSE vs Exact' + (' (log scale)' if cfg_rmse.get('log_scale', True) else ''));
        if cfg_rmse.get('log_scale', True): ax5.set_yscale('log')
        ax5.set_title(f'RMSE vs Time (Act: {activation_label})'); ax5.grid(True, which='both', ls=':'); ax5.legend(fontsize=cfg_rmse.get('legend_fontsize', 'small'));
        if cfg_rmse.get('ylim'): ax5.set_ylim(cfg_rmse['ylim'])
        if cfg_traj.get('time_xlim'): ax5.set_xlim(cfg_traj['time_xlim'])
        fig5.tight_layout(); plt.savefig(f'{filename_prefix}_error.png', dpi=dpi)
    except Exception as e: print(f"Error plotting RMSE: {e}")
    finally: plt.close(fig5) if fig5 else None

    # Plot 6: Total Loss History
    cfg_loss_total = plot_config.get('loss_total', {}); fig6=None
    try:
        fig6, ax6 = plt.subplots(1, 1, figsize=cfg_loss_total.get('figsize', (10, 6))); plot_loss=False
        for label, data in predictions_subset.items():
            hist=data.get('history'); s=styles.get(label)
            if s and hist and 'epoch' in hist and 'total' in hist and len(hist['epoch'])==len(hist['total']):
                 epochs=np.array(hist['epoch']); loss=np.array(hist['total']); valid=np.isfinite(loss) & (loss > 0 if cfg_loss_total.get('log_scale', True) else True)
                 if np.any(valid): ax6.plot(epochs[valid], loss[valid], label=label, **s); plot_loss=True
        if not plot_loss: print("    Skipping total loss plot - no valid data to plot."); plt.close(fig6); return
        ax6.set_xlabel('Epoch'); ax6.set_ylabel('Total Loss' + (' (log scale)' if cfg_loss_total.get('log_scale', True) else ''));
        if cfg_loss_total.get('log_scale', True): ax6.set_yscale('log')
        if cfg_loss_total.get('ylim'): ax6.set_ylim(cfg_loss_total['ylim'])
        ax6.set_title(f'Total Training Loss (Act: {activation_label})'); ax6.legend(fontsize=cfg_loss_total.get('legend_fontsize', 'small')); ax6.grid(True, which='both', ls=':'); fig6.tight_layout(); plt.savefig(f'{filename_prefix}_total_loss.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Loss History: {e}")
    finally: plt.close(fig6) if fig6 else None

    # Plot 7 & 8: 3D Phase Space Comparison
    cfg_3d = plot_config.get('3d_phase', {});
    try:
        skip_3d=max(1, num_points_plot // cfg_3d.get('skip_ratio', 500)); start_marker_size = cfg_3d.get('start_marker_size', 80)
        view_elev = cfg_3d.get('view_elev', 25.); view_azim = cfg_3d.get('view_azim', -55.); leg_fs = cfg_3d.get('legend_fontsize', 'x-small')
        init_st_ref = exact_data_highres.get('states'); init_st = init_st_ref[0] if init_st_ref is not None and len(init_st_ref) > 0 else None
        for p_idx, p_lbl in [(1,'p1'), (3,'p2')]:
            fig_3d=None 
            try:
                fig_3d=plt.figure(figsize=cfg_3d.get('figsize', (9, 9))); ax_3d=fig_3d.add_subplot(111,projection='3d'); has_3d_data = False
                for label, data in predictions_subset.items():
                    s=styles.get(label, styles.get('Exact')); st=data.get('states')
                    if st is not None and st.ndim == 2 and st.shape[1]==4:
                        valid=np.isfinite(st[::skip_3d,[0,2,p_idx]]).all(axis=1)
                        if np.any(valid): ax_3d.plot(st[::skip_3d,0][valid], st[::skip_3d,2][valid], st[::skip_3d,p_idx][valid], label=label, lw=s['lw']*0.7, alpha=s['alpha']*0.7, zorder=s.get('zorder',1), color=s['color'], ls=s.get('ls','-')); has_3d_data = True
                    else: print(f"    Warning: Invalid/missing 3D data for {label} ({p_lbl})")
                if not has_3d_data: print(f"    Skipping 3D plot ({p_lbl}) - no valid data."); plt.close(fig_3d); continue
                if init_st is not None: ax_3d.scatter(init_st[0],init_st[2],init_st[p_idx],c='darkviolet',marker='o',s=start_marker_size,label='Start',depthshade=False,zorder=100)
                ax_3d.set_xlabel('$q_1$'); ax_3d.set_ylabel('$q_2$'); ax_3d.set_zlabel(f'${p_lbl}$')
                ax_3d.set_title(f'3D Phase ($q_1,q_2,{p_lbl}$, Act: {activation_label})'); ax_3d.legend(fontsize=leg_fs); ax_3d.view_init(elev=view_elev,azim=view_azim); fig_3d.tight_layout()
                plt.savefig(f'{filename_prefix}_3d_phase_q1q2{p_lbl}.png', dpi=dpi)
            finally: plt.close(fig_3d) if fig_3d else None 
    except Exception as e: print(f"Error plotting 3D Phase Space: {e}")

# Plotting Function 2: Detailed Plots
def plot_single_model_details(model_label, model_pred_data, exact_data_highres, exact_data_eval, base_filename, plot_config):
    print(f"\n--- Plots: Detailed ({model_label}) ---")
    cfg_glob = plot_config.get('global', {}); dpi = cfg_glob.get('dpi', 300)
    default_skip_ratio = cfg_glob.get('plot_skip_ratio', 1000); phase_ms = cfg_glob.get('phase_marker_size', 1); poincare_ms = cfg_glob.get('poincare_marker_size', 4)
    if not model_pred_data or not exact_data_eval or not exact_data_highres: print("    Skipping detailed plot - missing prediction or exact data."); return
    if not all(k in model_pred_data for k in ['t_eval', 'states', 'H_norm', 'rmse_vs_time']): print(f"    Skipping detailed plot for {model_label} - missing essential data keys."); return

    # 
    model_color = 'firebrick' if 'Enhanced' in model_label else 'red' 

    plot_data = {'Exact': exact_data_eval, model_label: model_pred_data}
    styles = {'Exact': {'color':'black', 'ls':'-', 'lw':1, 'alpha':0.8, 'zorder':10},
              model_label: {'color': model_color, 'ls':'--', 'lw':1.2, 'alpha':0.9, 'zorder': 5}}

    t_eval = model_pred_data['t_eval']; num_points = len(t_eval); plot_skip = max(1, num_points // default_skip_ratio)
    filename_prefix = f"{base_filename}_Detail_{model_label.replace('_','-').replace('(','').replace(')','')}"

    # Plot 1: Trajectory
    cfg_traj = plot_config.get('trajectory', {}); fig1=None
    try:
        fig1, axs1 = plt.subplots(2, 1, figsize=cfg_traj.get('figsize', (10, 7)), sharex=True); plotted_any = False
        for lbl, data in plot_data.items():
             s=styles[lbl]; t=data.get('t_eval'); st=data.get('states')
             if t is not None and st is not None and st.ndim==2 and st.shape[1]>=3:
                 valid=np.isfinite(st[:,[0,2]]).all(axis=1)
                 if np.any(valid): axs1[0].plot(t[::plot_skip][valid[::plot_skip]], st[::plot_skip,0][valid[::plot_skip]], label=f'{lbl} $q_1$', **s); axs1[1].plot(t[::plot_skip][valid[::plot_skip]], st[::plot_skip,2][valid[::plot_skip]], label=f'{lbl} $q_2$', **s); plotted_any = True
        if not plotted_any: print("    Skipping detail trajectory plot - no valid data."); plt.close(fig1); return
        axs1[0].set_ylabel('$q_1$'); axs1[0].grid(True, ls=':'); axs1[0].legend(fontsize=cfg_traj.get('legend_fontsize', 'small'))
        if cfg_traj.get('q1_ylim'): axs1[0].set_ylim(cfg_traj['q1_ylim'])
        axs1[1].set_ylabel('$q_2$'); axs1[1].grid(True, ls=':'); axs1[1].set_xlabel('Time'); axs1[1].legend(fontsize=cfg_traj.get('legend_fontsize', 'small'))
        if cfg_traj.get('q2_ylim'): axs1[1].set_ylim(cfg_traj['q2_ylim'])
        if cfg_traj.get('time_xlim'): axs1[1].set_xlim(cfg_traj['time_xlim'])
        fig1.suptitle(f'Trajectory: {model_label} vs Exact'); fig1.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(f'{filename_prefix}_traj.png', dpi=dpi)
    except Exception as e: print(f"Error plotting detail traj: {e}")
    finally: plt.close(fig1) if fig1 else None

    # Plot 2: Phase Space
    cfg_phase = plot_config.get('phase', {}); fig2=None
    try:
        fig2, axs2 = plt.subplots(1, 2, figsize=cfg_phase.get('figsize', (12, 5))); plotted_any = False
        for lbl, data in plot_data.items():
             s=styles[lbl]; st=data.get('states')
             if st is not None and st.ndim == 2 and st.shape[1]==4:
                 valid=np.isfinite(st).all(axis=1)
                 if np.any(valid): axs2[0].plot(st[::plot_skip,0][valid[::plot_skip]], st[::plot_skip,1][valid[::plot_skip]], '.', ms=phase_ms, label=lbl, color=s['color'], alpha=s['alpha'], zorder=s.get('zorder',1)); axs2[1].plot(st[::plot_skip,2][valid[::plot_skip]], st[::plot_skip,3][valid[::plot_skip]], '.', ms=phase_ms, label=lbl, color=s['color'], alpha=s['alpha'], zorder=s.get('zorder',1)); plotted_any = True
        if not plotted_any: print("    Skipping detail phase plot - no valid data."); plt.close(fig2); return
        axs2[0].set_xlabel('$q_1$'); axs2[0].set_ylabel('$p_1$'); axs2[0].set_title('$(q_1, p_1)$'); axs2[0].grid(True, ls=':'); axs2[0].legend(fontsize=cfg_phase.get('legend_fontsize', 'small'), markerscale=cfg_phase.get('legend_markerscale', 5));
        if cfg_phase.get('q1p1_xlim'): axs2[0].set_xlim(cfg_phase['q1p1_xlim'])
        if cfg_phase.get('q1p1_ylim'): axs2[0].set_ylim(cfg_phase['q1p1_ylim'])
        if cfg_phase.get('equal_aspect', True): axs2[0].set_aspect('equal', adjustable='box')
        axs2[1].set_xlabel('$q_2$'); axs2[1].set_ylabel('$p_2$'); axs2[1].set_title('$(q_2, p_2)$'); axs2[1].grid(True, ls=':'); axs2[1].legend(fontsize=cfg_phase.get('legend_fontsize', 'small'), markerscale=cfg_phase.get('legend_markerscale', 5));
        if cfg_phase.get('q2p2_xlim'): axs2[1].set_xlim(cfg_phase['q2p2_xlim'])
        if cfg_phase.get('q2p2_ylim'): axs2[1].set_ylim(cfg_phase['q2p2_ylim'])
        if cfg_phase.get('equal_aspect', True): axs2[1].set_aspect('equal', adjustable='box')
        fig2.suptitle(f'Phase Space: {model_label} vs Exact'); fig2.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(f'{filename_prefix}_phase.png', dpi=dpi)
    except Exception as e: print(f"Error plotting detail phase: {e}")
    finally: plt.close(fig2) if fig2 else None

    # Plot 3: Energy
    cfg_energy = plot_config.get('energy', {}); fig3=None
    try:
        fig3, ax3 = plt.subplots(1, 1, figsize=cfg_energy.get('figsize', (10, 5))); min_E, max_E = 1.0, 1.0; has_valid_energy = False
        for lbl, data in plot_data.items():
            s=styles[lbl]; h_norm=data.get('H_norm'); t=data.get('t_eval')
            if h_norm is not None and t is not None:
                valid=np.isfinite(h_norm)
                if np.any(valid): ax3.plot(t[valid], h_norm[valid], label=lbl, **s); min_E = min(min_E, np.nanmin(h_norm[valid])); max_E = max(max_E, np.nanmax(h_norm[valid])); has_valid_energy = True
        if not has_valid_energy: print("    Skipping detail energy plot - no valid data."); plt.close(fig3); return
        ax3.set_xlabel('Time'); ax3.set_ylabel('Norm. Energy (H/H$_0$)'); ax3.set_title(f'Energy: {model_label} vs Exact'); ax3.grid(True,ls=':'); ax3.legend(fontsize=cfg_energy.get('legend_fontsize', 'small'))
        force_ylim = cfg_energy.get('force_ylim');
        if force_ylim: ax3.set_ylim(force_ylim)
        else: rel_range = cfg_energy.get('ylim_rel_range', 1.3); energy_range = max(max_E - 1.0, 1.0 - min_E, 0.01); ax3.set_ylim(1.0 - energy_range * rel_range, 1.0 + energy_range * rel_range)
        if cfg_traj.get('time_xlim'): ax3.set_xlim(cfg_traj['time_xlim'])
        fig3.tight_layout(); plt.savefig(f'{filename_prefix}_energy.png', dpi=dpi)
    except Exception as e: print(f"Error plotting detail energy: {e}")
    finally: plt.close(fig3) if fig3 else None

    # Plot 4: Poincaré
    cfg_poincare = plot_config.get('poincare', {}); fig4=None
    try:
        fig4, ax4 = plt.subplots(1, 1, figsize=cfg_poincare.get('figsize', (7, 7))); print(f"  Poincaré ({model_label})..."); plotted_any = False
        t_hr, s_hr = exact_data_highres.get('t'), exact_data_highres.get('states'); qe_hr, pe_hr = ([], []);
        if t_hr is not None and s_hr is not None: qe_hr, pe_hr = compute_poincare(t_hr, s_hr)
        t_m, s_m = model_pred_data.get('t_eval'), model_pred_data.get('states'); qm, pm = ([], []);
        if t_m is not None and s_m is not None: qm, pm = compute_poincare(t_m, s_m)
        plotted_exact = False
        if len(qe_hr) > 0: se=styles['Exact']; ax4.plot(qe_hr,pe_hr,'.',ms=max(1, poincare_ms // 2),label=f'Exact ({len(qe_hr)}pts)',color=se['color'], alpha=se['alpha']*0.7, zorder=se.get('zorder', 10)-1); plotted_exact=True; plotted_any=True; qmn_e, qmx_e = np.min(qe_hr), np.max(qe_hr); pmn_e, pmx_e = np.min(pe_hr), np.max(pe_hr); qr_e = max(qmx_e - qmn_e, 1e-2); pr_e = max(pmx_e - pmn_e, 1e-2); default_xlim = (qmn_e - 0.1 * qr_e, qmx_e + 0.1 * qr_e); default_ylim = (pmn_e - 0.1 * pr_e, pmx_e + 0.1 * pr_e); ax4.set_xlim(cfg_poincare.get('xlim', default_xlim)); ax4.set_ylim(cfg_poincare.get('ylim', default_ylim))
        if len(qm)>0: sm=styles[model_label]; ax4.plot(qm,pm,'.',ms=poincare_ms,label=f'{model_label} ({len(qm)}pts)',color=sm['color'],alpha=sm['alpha'], zorder=sm.get('zorder', 5)); plotted_any=True
        if not plotted_any: print("    Skipping detail Poincaré plot - no valid data."); plt.close(fig4); return
        ax4.set_xlabel('$q_2$'); ax4.set_ylabel('$p_2$'); ax4.set_title(f'Poincaré ($q_1=0, p_1>0$): {model_label} vs Exact'); ax4.legend(fontsize=cfg_poincare.get('legend_fontsize', 'small'), markerscale=cfg_poincare.get('legend_markerscale', 3)); ax4.grid(True,ls=':');
        if cfg_poincare.get('equal_aspect', True): ax4.set_aspect('equal', adjustable='box')
        fig4.tight_layout(); plt.savefig(f'{filename_prefix}_poincare.png', dpi=dpi)
    except Exception as e: print(f"Error plotting detail poincare: {e}")
    finally: plt.close(fig4) if fig4 else None

    # Plot 5: Loss History 
    cfg_loss_detail = plot_config.get('loss_detail', {}); fig5=None
    history = model_pred_data.get('history')
    if history and 'epoch' in history and len(history['epoch']) > 0:
        try:
            fig5, ax5 = plt.subplots(1, 1, figsize=cfg_loss_detail.get('figsize', (11, 6))); plot_loss = False
            epochs = np.array(history['epoch']); loss_keys = ['total','ode','ic','energy','energy_grad','symplectic','l2']
            colors = plt.cm.tab10(np.linspace(0, 1, len(loss_keys)))
            for i, key in enumerate(loss_keys):
                if key in history and len(history[key]) == len(epochs):
                    loss = np.array(history[key]); valid = np.isfinite(loss) & (loss > 1e-12 if cfg_loss_detail.get('log_scale', True) else True)
                    if np.any(valid): ax5.plot(epochs[valid], loss[valid], label=key.capitalize(), color=colors[i], lw= 2 if key=='total' else 1.2, alpha=0.9); plot_loss = True
            if not plot_loss: print("    Skipping detailed loss plot - no valid data."); plt.close(fig5); return
            ax5.set_xlabel('Epoch'); ax5.set_ylabel('Loss Value' + (' (log scale)' if cfg_loss_detail.get('log_scale', True) else ''))
            if cfg_loss_detail.get('log_scale', True): ax5.set_yscale('log')
            if cfg_loss_detail.get('ylim'): ax5.set_ylim(cfg_loss_detail['ylim'])
            ax5.set_title(f'Detailed Training Loss History ({model_label})'); ax5.legend(fontsize=cfg_loss_detail.get('legend_fontsize', 'small'), bbox_to_anchor=cfg_loss_detail.get('legend_bbox', (1.02, 1.0)), loc=cfg_loss_detail.get('legend_loc', "upper left")); ax5.grid(True, which='both', linestyle=':')
            fig5.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(f'{filename_prefix}_loss_detail.png', dpi=dpi)
        except Exception as e: print(f"Error plotting detail loss: {e}")
        finally: plt.close(fig5) if fig5 else None
    else: print("    Skipping detailed loss plot - no history found.")

    # Plot 6: Heatmap of Loss Components
    cfg_heatmap = plot_config.get('loss_heatmap', {}); fig6=None
    if history and 'epoch' in history and len(history['epoch']) > 0:
        try:
            fig6, ax6 = plt.subplots(figsize=cfg_heatmap.get('figsize', (8, 6))); heatmap_data = []; plotted_keys = []
            loss_keys = ['ode','ic','energy','energy_grad','symplectic','l2']; epochs = history['epoch']; valid_epochs = np.array(epochs); heatmap_matrix_list = []
            for key in loss_keys:
                 loss_hist = history.get(key, [])
                 if len(loss_hist) == len(epochs): loss_arr = np.array(loss_hist); loss_arr[~np.isfinite(loss_arr)] = np.nan; heatmap_matrix_list.append(loss_arr); plotted_keys.append(key)
            if not heatmap_matrix_list: print("    Skipping loss heatmap - no valid loss components found."); plt.close(fig6); return
            heatmap_matrix = np.stack(heatmap_matrix_list, axis=0); log_loss = np.log10(np.abs(heatmap_matrix) + 1e-12); log_loss[np.isnan(heatmap_matrix)] = np.nan
            num_epochs_total = len(epochs); ytick_step = max(1, num_epochs_total // 10); yticklabels = valid_epochs[::ytick_step]
            sns.heatmap(log_loss.T, ax=ax6, cmap=cfg_heatmap.get('cmap', 'viridis'), cbar=True, xticklabels=plotted_keys, yticklabels=yticklabels, cbar_kws={'label':'log10(|Loss|)'}, mask=np.isnan(log_loss.T))
            ax6.set_title(f'Loss Component Heatmap ({model_label})'); ax6.set_xlabel('Loss Component'); ax6.set_ylabel('Epoch'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=8, rotation=0);
            fig6.tight_layout(); plt.savefig(f'{filename_prefix}_loss_heatmap.png', dpi=dpi)
        except Exception as e: print(f"Error plotting detail heatmap: {e}")
        finally: plt.close(fig6) if fig6 else None
    else: print("    Skipping loss heatmap - no history found.")

    # Plot 7 & 8: 3D Phase Space
    cfg_3d = plot_config.get('3d_phase', {}); skip_3d=max(1, num_points // cfg_3d.get('skip_ratio', 500)); start_marker_size = cfg_3d.get('start_marker_size', 80)
    view_elev = cfg_3d.get('view_elev', 25.); view_azim = cfg_3d.get('view_azim', -55.); leg_fs = cfg_3d.get('legend_fontsize', 'x-small')
    init_st_ref = exact_data_highres.get('states'); init_st = init_st_ref[0] if init_st_ref is not None and len(init_st_ref) > 0 else None
    try:
        plot_data_3d = {'Exact_HighRes': {'states': exact_data_highres.get('states'), 't_eval': exact_data_highres.get('t')}, model_label: model_pred_data}
        #
        model_color_3d = 'firebrick' if 'Enhanced' in model_label else 'blue' # 

        styles_3d = {'Exact_HighRes': {'color':'black', 'ls':'-', 'lw':1, 'alpha':0.6, 'zorder':10},
                     model_label: {'color': model_color_3d, 'ls':'--', 'lw':1.3, 'alpha':0.8, 'zorder': 5}}

        for p_idx, p_lbl in [(1,'p1'), (3,'p2')]:
            fig_3d=None # Reset fig handle
            try:
                fig_3d=plt.figure(figsize=cfg_3d.get('figsize', (9, 9))); ax_3d=fig_3d.add_subplot(111,projection='3d'); has_3d_data = False
                for label, data in plot_data_3d.items():
                    s=styles_3d[label]; st=data.get('states')
                    if st is None or st.ndim != 2 or st.shape[1] != 4: print(f"    Warning: Invalid/missing 3D states for {label}"); continue
                    current_skip = skip_3d if label == model_label else max(1, len(st) // cfg_3d.get('skip_ratio', 500))
                    st_plot = st[::current_skip,:]; valid=np.isfinite(st_plot[:,[0,2,p_idx]]).all(axis=1)
                    if np.any(valid): ax_3d.plot(st_plot[valid,0], st_plot[valid,2], st_plot[valid,p_idx], label=label.replace('_HighRes',''), lw=s['lw'], alpha=s['alpha'], color=s['color'], ls=s['ls'], zorder=s.get('zorder',1)); has_3d_data = True
                if not has_3d_data: print(f"    Skipping detail 3D plot ({p_lbl}) - no valid data."); plt.close(fig_3d); continue
                if init_st is not None: ax_3d.scatter(init_st[0],init_st[2],init_st[p_idx],c='darkviolet',marker='o',s=start_marker_size,label='Start',depthshade=False,zorder=100)
                ax_3d.set_xlabel('$q_1$'); ax_3d.set_ylabel('$q_2$'); ax_3d.set_zlabel(f'${p_lbl}$'); ax_3d.set_title(f'3D Phase ($q_1,q_2,{p_lbl}$): {model_label} vs Exact'); ax_3d.legend(fontsize=leg_fs); ax_3d.view_init(elev=view_elev,azim=view_azim); fig_3d.tight_layout()
                plt.savefig(f'{filename_prefix}_3d_phase_q1q2{p_lbl}.png', dpi=dpi)
            finally: plt.close(fig_3d) if fig_3d else None
    except Exception as e: print(f"Error plotting detail 3D: {e}")

    print(f"--- Finished Detailed Plots for: {model_label} ---")


# --- Euler vs EnhancedSympNet Plot ---
def plot_euler_vs_enhancedsympnet(euler_data, enhanced_data, exact_data_eval, exact_data_highres, enhanced_label, base_filename, plot_config):
    print(f"\n--- Plots: Euler vs {enhanced_label} ---")
    cfg_glob = plot_config.get('global', {}); dpi = cfg_glob.get('dpi', 300)
    default_skip_ratio = cfg_glob.get('plot_skip_ratio', 1000); phase_ms = cfg_glob.get('phase_marker_size', 1); poincare_ms = cfg_glob.get('poincare_marker_size', 4)
    if not euler_data or not enhanced_data or not exact_data_eval or not exact_data_highres: print("    Skipping Euler vs Enhanced plot - missing required data."); return
    plot_data = {'Exact': exact_data_eval, 'Euler': euler_data, enhanced_label: enhanced_data}
    if not all(plot_data.get(k) for k in ['Exact', 'Euler', enhanced_label]): print("    Error: Missing data structures for Euler vs EnhancedSympNet plot."); return


    styles = {
        'Exact': {'color': 'black', 'ls': '-', 'lw': 1.0, 'alpha': 0.9, 'zorder': 50},
        'Euler': {'color': 'darkseagreen', 'ls': '-.', 'lw': 1.3, 'alpha': 0.7, 'zorder': 10},
        enhanced_label: {'color': 'firebrick', 'ls': '--', 'lw': 1.5, 'alpha': 0.8, 'zorder': 40}
    }

    ref_data = plot_data['Exact']; num_points_plot = len(ref_data['t_eval']) if ref_data and 't_eval' in ref_data else 0
    if num_points_plot == 0: print("   Skipping plot - reference data has no time points."); return
    plot_skip = max(1, num_points_plot // default_skip_ratio)
    enhanced_label_safe = enhanced_label.replace('_','-').replace('(','').replace(')','')
    filename_prefix = f"{base_filename}_Compare_Euler_vs_{enhanced_label_safe}"

    # Plot 1: Trajectories
    cfg_traj = plot_config.get('trajectory', {}); fig1=None
    try:
        fig1, axs1 = plt.subplots(2, 1, figsize=cfg_traj.get('figsize', (12, 8)), sharex=True); plotted_any=False
        for label, data in plot_data.items():
            s=styles[label]; t=data.get('t_eval'); st=data.get('states')
            if t is not None and st is not None and st.ndim == 2 and st.shape[1]>=3:
                valid=np.isfinite(st[:,0]) & np.isfinite(st[:,2])
                if np.any(valid): axs1[0].plot(t[::plot_skip][valid[::plot_skip]], st[::plot_skip,0][valid[::plot_skip]], label=f'{label} $q_1$', **s); axs1[1].plot(t[::plot_skip][valid[::plot_skip]], st[::plot_skip,2][valid[::plot_skip]], label=f'{label} $q_2$', **s); plotted_any=True
        if not plotted_any: print("    Skipping Euler/Enhanced trajectory plot - no valid data."); plt.close(fig1); return
        axs1[0].set_ylabel('$q_1$'); axs1[0].grid(True, ls=':'); axs1[0].legend(fontsize=cfg_traj.get('legend_fontsize', 8), loc='best');
        if cfg_traj.get('q1_ylim'): axs1[0].set_ylim(cfg_traj['q1_ylim'])
        axs1[1].set_ylabel('$q_2$'); axs1[1].grid(True, ls=':'); axs1[1].set_xlabel('Time'); axs1[1].legend(fontsize=cfg_traj.get('legend_fontsize', 8), loc='best');
        if cfg_traj.get('q2_ylim'): axs1[1].set_ylim(cfg_traj['q2_ylim'])
        if cfg_traj.get('time_xlim'): axs1[1].set_xlim(cfg_traj['time_xlim'])
        fig1.suptitle(f'Trajectories: Euler vs {enhanced_label}'); fig1.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(f'{filename_prefix}_traj.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Euler vs Enhanced traj: {e}")
    finally: plt.close(fig1) if fig1 else None

    # Plot 2: Phase Space
    cfg_phase = plot_config.get('phase', {}); fig2=None
    try:
        fig2, axs2 = plt.subplots(1, 2, figsize=cfg_phase.get('figsize', (14, 6))); plotted_any=False
        for label, data in plot_data.items():
            s=styles[label]; st=data.get('states')
            if st is not None and st.ndim == 2 and st.shape[1]==4:
                valid = np.isfinite(st).all(axis=1); plot_ms = phase_ms * 1.5 if label != 'Exact' else phase_ms; plot_alpha = s['alpha'] * (0.8 if label == 'Exact' else 1.0)
                if np.any(valid): axs2[0].plot(st[::plot_skip,0][valid[::plot_skip]], st[::plot_skip,1][valid[::plot_skip]], '.', ms=plot_ms, label=label, color=s['color'], alpha=plot_alpha, zorder=s.get('zorder',1)); axs2[1].plot(st[::plot_skip,2][valid[::plot_skip]], st[::plot_skip,3][valid[::plot_skip]], '.', ms=plot_ms, label=label, color=s['color'], alpha=plot_alpha, zorder=s.get('zorder',1)); plotted_any=True
        if not plotted_any: print("    Skipping Euler/Enhanced phase plot - no valid data."); plt.close(fig2); return
        axs2[0].set_xlabel('$q_1$'); axs2[0].set_ylabel('$p_1$'); axs2[0].set_title('$(q_1, p_1)$'); axs2[0].grid(True, ls=':'); axs2[0].legend(fontsize=cfg_phase.get('legend_fontsize', 7), markerscale=cfg_phase.get('legend_markerscale', 5));
        if cfg_phase.get('q1p1_xlim'): axs2[0].set_xlim(cfg_phase['q1p1_xlim'])
        if cfg_phase.get('q1p1_ylim'): axs2[0].set_ylim(cfg_phase['q1p1_ylim'])
        if cfg_phase.get('equal_aspect', True): axs2[0].set_aspect('equal', adjustable='box')
        axs2[1].set_xlabel('$q_2$'); axs2[1].set_ylabel('$p_2$'); axs2[1].set_title('$(q_2, p_2)$'); axs2[1].grid(True, ls=':'); axs2[1].legend(fontsize=cfg_phase.get('legend_fontsize', 7), markerscale=cfg_phase.get('legend_markerscale', 5));
        if cfg_phase.get('q2p2_xlim'): axs2[1].set_xlim(cfg_phase['q2p2_xlim'])
        if cfg_phase.get('q2p2_ylim'): axs2[1].set_ylim(cfg_phase['q2p2_ylim'])
        if cfg_phase.get('equal_aspect', True): axs2[1].set_aspect('equal', adjustable='box')
        fig2.suptitle(f'Phase Space: Euler vs {enhanced_label}'); fig2.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(f'{filename_prefix}_phase.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Euler vs Enhanced phase: {e}")
    finally: plt.close(fig2) if fig2 else None

    # Plot 3: Energy
    cfg_energy = plot_config.get('energy', {}); fig3=None
    try:
        fig3, ax3 = plt.subplots(1, 1, figsize=cfg_energy.get('figsize', (10, 5))); min_E, max_E = 1.0, 1.0; has_valid_energy = False
        for label, data in plot_data.items():
            s=styles[label]; h_norm=data.get('H_norm'); t=data.get('t_eval')
            if h_norm is not None and t is not None:
                valid=np.isfinite(h_norm)
                if np.any(valid): ax3.plot(t[valid], h_norm[valid], label=label, **s); min_E = min(min_E, np.nanmin(h_norm[valid])); max_E = max(max_E, np.nanmax(h_norm[valid])); has_valid_energy = True
        if not has_valid_energy: print("    Skipping Euler/Enhanced energy plot - no valid data."); plt.close(fig3); return
        ax3.set_xlabel('Time'); ax3.set_ylabel('Norm. Energy (H/H$_0$)'); ax3.set_title(f'Energy: Euler vs {enhanced_label}'); ax3.grid(True, ls=':'); ax3.legend(fontsize=cfg_energy.get('legend_fontsize', 'small'))
        force_ylim = cfg_energy.get('force_ylim');
        if force_ylim: ax3.set_ylim(force_ylim)
        else: rel_range = cfg_energy.get('ylim_rel_range', 1.3); energy_range = max(max_E - 1.0, 1.0 - min_E, 0.01); ax3.set_ylim(1.0 - energy_range * rel_range, 1.0 + energy_range * rel_range)
        if cfg_traj.get('time_xlim'): ax3.set_xlim(cfg_traj['time_xlim'])
        fig3.tight_layout(); plt.savefig(f'{filename_prefix}_energy.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Euler vs Enhanced energy: {e}")
    finally: plt.close(fig3) if fig3 else None

    # Plot 4: Poincaré
    cfg_poincare = plot_config.get('poincare', {}); fig4=None
    try:
        fig4, ax4 = plt.subplots(1, 1, figsize=cfg_poincare.get('figsize', (8, 8))); print(f"  Poincaré (Euler vs {enhanced_label})..."); plotted_any=False
        poincare_data = {}; t_hr, s_hr = exact_data_highres.get('t'), exact_data_highres.get('states'); qe_hr, pe_hr = ([], []);
        if t_hr is not None and s_hr is not None: qe_hr, pe_hr = compute_poincare(t_hr, s_hr); poincare_data['Exact_HighRes'] = (qe_hr, pe_hr)
        for lbl, d in plot_data.items():
            if lbl.lower() != 'exact': t_ev, s_ev = d.get('t_eval'), d.get('states');
            if t_ev is not None and s_ev is not None: poincare_data[lbl] = compute_poincare(t_ev, s_ev)
        plotted_exact = False
        if 'Exact_HighRes' in poincare_data:
            qe_hr, pe_hr = poincare_data['Exact_HighRes']
            if len(qe_hr) > 0: se = styles['Exact']; ax4.plot(qe_hr, pe_hr, '.', ms=max(1, poincare_ms // 2), label='Exact', color=se['color'], alpha=se['alpha']*0.7, zorder=se.get('zorder', 50)-1); plotted_exact = True; plotted_any=True; qmn_e, qmx_e = np.min(qe_hr), np.max(qe_hr); pmn_e, pmx_e = np.min(pe_hr), np.max(pe_hr); qr_e = max(qmx_e - qmn_e, 1e-2); pr_e = max(pmx_e - pmn_e, 1e-2); default_xlim = (qmn_e - 0.1 * qr_e, qmx_e + 0.1 * qr_e); default_ylim = (pmn_e - 0.1 * pr_e, pmx_e + 0.1 * pr_e); ax4.set_xlim(cfg_poincare.get('xlim', default_xlim)); ax4.set_ylim(cfg_poincare.get('ylim', default_ylim))
        for label, data in poincare_data.items():
            if label != 'Exact_HighRes' and label in styles:
                q,p = data
                if len(q) > 0: s = styles[label]; ax4.plot(q, p, '.', ms=poincare_ms, label=f'{label} ({len(q)}pts)', color=s['color'], alpha=s['alpha'], zorder=s.get('zorder', 1)); plotted_any=True
        if not plotted_any: print("    Skipping Euler/Enhanced Poincaré plot - no valid data."); plt.close(fig4); return
        ax4.set_xlabel('$q_2$'); ax4.set_ylabel('$p_2$'); ax4.set_title(f'Poincaré: Euler vs {enhanced_label}'); ax4.legend(fontsize=cfg_poincare.get('legend_fontsize', 'small'), markerscale=cfg_poincare.get('legend_markerscale', 3)); ax4.grid(True, ls=':');
        if cfg_poincare.get('equal_aspect', True): ax4.set_aspect('equal', adjustable='box')
        fig4.tight_layout(); plt.savefig(f'{filename_prefix}_poincare.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Euler vs Enhanced Poincare: {e}")
    finally: plt.close(fig4) if fig4 else None

    # Plot 5: RMSE
    cfg_rmse = plot_config.get('rmse', {}); fig5=None
    try:
        fig5, ax5 = plt.subplots(1, 1, figsize=cfg_rmse.get('figsize', (10, 5))); has_rmse = False
        for label, data in plot_data.items():
            if label.lower() != 'exact':
                s=styles[label]; rmse = data.get('rmse_vs_time'); t=data.get('t_eval')
                if rmse is not None and t is not None:
                    valid=np.isfinite(rmse)
                    if np.any(valid): ax5.plot(t[valid], rmse[valid], label=label, **s); has_rmse = True
        if not has_rmse: print("    Skipping Euler vs Enhanced RMSE plot - no valid data."); plt.close(fig5); return
        ax5.set_xlabel('Time'); ax5.set_ylabel('RMSE vs Exact' + (' (log scale)' if cfg_rmse.get('log_scale', True) else ''));
        if cfg_rmse.get('log_scale', True): ax5.set_yscale('log')
        ax5.set_title(f'RMSE vs Time: Euler vs {enhanced_label}'); ax5.grid(True, which='both', ls=':'); ax5.legend(fontsize=cfg_rmse.get('legend_fontsize', 'small'));
        if cfg_rmse.get('ylim'): ax5.set_ylim(cfg_rmse['ylim'])
        if cfg_traj.get('time_xlim'): ax5.set_xlim(cfg_traj['time_xlim'])
        fig5.tight_layout(); plt.savefig(f'{filename_prefix}_error.png', dpi=dpi)
    except Exception as e: print(f"Error plotting Euler vs Enhanced RMSE: {e}")
    finally: plt.close(fig5) if fig5 else None

    print(f"--- Finished Euler vs {enhanced_label} Plots ---")


# --- Central Plotting Control Function ---
def generate_all_plots(all_predictions, exact_data_highres, initial_state_np, initial_energy, dynamics_label, t_span_end, base_filename, plot_config, activations_to_test):
    """Generates all comparison and detail plots based on the provided configuration."""
    print("\n--- Generating All Plots ---")

    exact_data_eval = all_predictions.get('Exact')
    if not exact_data_eval:
        print("Warning: 'Exact' data (on evaluation grid) not found. Some plots might be missing reference.")
        dummy_t = np.linspace(0, t_span_end, 2)
        exact_data_eval = {'states': np.full((2, 4), np.nan), 'H_norm': np.full(2, np.nan), 't_eval': dummy_t, 'rmse_vs_time': np.full(2, np.nan), 'history': {}}

    # 1. Generate Comparison Plots for Each Activation
    for act in activations_to_test:
        subset_preds = {}
        if 'Exact' in all_predictions: subset_preds['Exact'] = all_predictions['Exact']
        if 'Euler' in all_predictions: subset_preds['Euler'] = all_predictions['Euler']
        for model_type in ['EnhancedSympNet', 'SimplePINN', 'SimpleHNN']:
            label = f"{model_type}_{act}"
            if label in all_predictions: subset_preds[label] = all_predictions[label]
        if len(subset_preds) > 2: # Need Exact + Euler + 1 model minimum
             plot_comparison_set(subset_preds, exact_data_highres, act, base_filename, plot_config)
        elif len(subset_preds) > 0:
             print(f"Skipping comparison plot for activation '{act}': Not enough models found (found {len(subset_preds)} including Exact/Euler).")

    # 2. Generate Detailed Plots for Each Individual Trained Model Run
    for label, pred_data in all_predictions.items():
        if label.lower() not in ['exact', 'euler']: # Only for learned models
            plot_single_model_details(label, pred_data, exact_data_highres, exact_data_eval, base_filename, plot_config)

    # 3. Generate Specific Euler vs EnhancedSympNet Plots
    if 'Euler' in all_predictions:
        euler_data = all_predictions['Euler']
        for act in activations_to_test:
            enhanced_label = f"EnhancedSympNet_{act}"
            if enhanced_label in all_predictions:
                enhanced_data = all_predictions[enhanced_label]
                plot_euler_vs_enhancedsympnet(euler_data, enhanced_data, exact_data_eval, exact_data_highres, enhanced_label, base_filename, plot_config)
            else:
                print(f"Skipping Euler vs Enhanced plot for {enhanced_label}: Model data not found.")
    else:
        print("Skipping Euler vs Enhanced plots: Euler data not found.")

    print("\n--- All Plot Generation Complete ---")

# --- Excel Export Function ---
def sanitize_sheet_name(name):
    """Removes invalid characters and truncates name for Excel sheet names."""
    name = re.sub(r'[\\/*?:\[\]]', '_', name)
    return name[:31]

def save_results_to_excel(all_predictions, base_filename):
    """Saves the time series data (t, q, p, H, RMSE) for all methods to an Excel file."""
    excel_filename = f"{base_filename}_results.xlsx"
    print(f"\n--- Saving results to Excel: {excel_filename} ---")
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            for label, data in all_predictions.items():
                sanitized_label = sanitize_sheet_name(label)
                print(f"  Processing sheet: {sanitized_label} (from {label})")
                if not isinstance(data, dict): print(f"    Warning: Skipping sheet '{sanitized_label}'. Data is not a dictionary."); continue
                if not all(k in data for k in ['t_eval', 'states', 'H_norm', 'rmse_vs_time']): print(f"    Warning: Skipping sheet '{sanitized_label}'. Missing data keys."); continue
                t = data['t_eval']; states = data['states']; h_norm = data['H_norm']; rmse = data['rmse_vs_time']
                if not isinstance(t, np.ndarray) or not isinstance(states, np.ndarray) or not isinstance(h_norm, np.ndarray) or not isinstance(rmse, np.ndarray): print(f"    Warning: Skipping sheet '{sanitized_label}'. Data components are not all numpy arrays."); continue
                n_points = len(t)
                if states.shape != (n_points, 4) or h_norm.shape != (n_points,) or rmse.shape != (n_points,): print(f"    Warning: Skipping sheet '{sanitized_label}'. Inconsistent data lengths/shapes."); continue
                df_data = {'Time': t, 'q1': states[:, 0], 'p1': states[:, 1], 'q2': states[:, 2], 'p2': states[:, 3], 'H_norm': h_norm, 'RMSE_vs_Exact': rmse }
                df = pd.DataFrame(df_data)
                print(f"    Saving DataFrame with {len(df)} rows to sheet: '{sanitized_label}'")
                df.to_excel(writer, sheet_name=sanitized_label, index=False, float_format="%.6e")
        print(f"--- Results successfully saved to {excel_filename} ---")
    except ImportError: print("\nError: Could not save to Excel. 'pandas' and/or 'openpyxl' required."); print("Install them ('pip install pandas openpyxl') and try again.")
    except Exception as e: print(f"\nAn unexpected error occurred saving results to Excel file '{excel_filename}': {e}")


# === Main Execution ===
if __name__ == "__main__":

    # --- Initial Conditions & Dynamics ---
    energy_level = 1/8; q1_0=0.0; q2_0=0.1; p2_0=0.0; p1_0=None; dynamics_label="Mixed_E0p125"
    print(f"Running simulation for: {dynamics_label} (Target E≈{energy_level:.4f})")

    # Calculate Missing Momentum Component
    V0 = 0.5*(q1_0**2+q2_0**2) + q1_0**2*q2_0 - (1/3)*q2_0**3
    target_ke = energy_level - V0
    if target_ke < -1e-9: raise ValueError(f"V0={V0:.4f} exceeds E={energy_level:.4f}.")
    target_ke = max(0, target_ke)
    if p1_0 is None and p2_0 is not None: p1_0_sq = 2*target_ke - p2_0**2; p1_0 = np.sqrt(max(0, p1_0_sq)) if p1_0_sq>=-1e-9 else float('nan')
    elif p2_0 is None and p1_0 is not None: p2_0_sq = 2*target_ke - p1_0**2; p2_0 = np.sqrt(max(0, p2_0_sq)) if p2_0_sq>=-1e-9 else float('nan')
    elif p1_0 is None and p2_0 is None: p1_0 = np.sqrt(2*target_ke); p2_0 = 0.0; print("Warning: Initialized p1=sqrt(2*KE), p2=0.")
    if np.isnan(p1_0) or np.isnan(p2_0): raise ValueError("Cannot achieve target energy with specified coordinates.")
    initial_state_np = np.array([q1_0, p1_0, q2_0, p2_0])
    initial_energy_check = henon_heiles_hamiltonian(torch.tensor(initial_state_np, dtype=torch.float64)).item()
    print(f"Initial State: q1={initial_state_np[0]:.4f}, p1={initial_state_np[1]:.4f}, q2={initial_state_np[2]:.4f}, p2={initial_state_np[3]:.4f}")
    print(f"Calculated Initial Energy H0 = {initial_energy_check:.6f}")
    if abs(initial_energy_check - energy_level) > 1e-5: print(f"Warning: Calculated H0 differs significantly from target E ({energy_level:.4f}).")

    # --- Simulation & Training Parameters ---
    t_span_end = 20.0          
    num_points_train = 4000   
    num_points_eval = 40000    
    num_points_benchmark = 300000 
    epochs = 20              
    batch_size = 32
    lr = 3e-4
    patience = 20             

    # --- Generate High-Resolution Benchmark ---
    t_exact_hr, states_exact_hr, H_exact_hr = exact_henon_heiles_solution(initial_state_np, t_span_end, num_points_benchmark)
    exact_data_highres = {'t': t_exact_hr, 'states': states_exact_hr, 'H': H_exact_hr}

    # --- Model Setup ---
    models_to_run = {}
    hidden_dim = 128; base_scale = 0.005;
 
    beta_adaptive = 0.1
    activations_to_test = ['tanh'] # Test only tanh first

    models_to_run['Euler'] = {'model': EulerIntegrator(), 'history': None}
    for act in activations_to_test:
        
        models_to_run[f"EnhancedSympNet_{act}"] = {'model': EnhancedSympNet(hidden_dim, act, base_scale, beta_adaptive)}
        models_to_run[f"SimplePINN_{act}"] = {'model': SimplePINN_ODEFunc(hidden_dim, act)}
        models_to_run[f"SimpleHNN_{act}"] = {'model': SimpleHNN_ODEFunc(hidden_dim, act)}

    # --- Train Models ---
    trained_models_data = {}
    trained_models_data['Euler'] = models_to_run['Euler']
    for label, data in models_to_run.items():
        if isinstance(data.get('model'), nn.Module):
            history = train_model(data['model'], initial_state_np, t_span_end, num_points_train, epochs, batch_size, lr, patience)
            trained_models_data[label] = {'model': data['model'].to('cpu'), 'history': history}
            if DEVICE=='cuda': torch.cuda.empty_cache()
        elif label != 'Euler': print(f"Skipping training for non-trainable: {label}")

    # --- Generate All Predictions ---
    all_predictions = generate_predictions(trained_models_data, exact_data_highres, initial_state_np, t_span_end, num_points_eval)

    # --- Define Plot Configuration ---
    plot_config = {
        'global': {'dpi': 150, 'plot_skip_ratio': 500, 'phase_marker_size': 1, 'poincare_marker_size': 3}, 
        'trajectory': {'figsize': (10, 6), 'q1_ylim': None, 'q2_ylim': None, 'time_xlim': None, 'legend_fontsize': 8},
        'phase': {'figsize': (12, 5), 'q1p1_xlim': None, 'q1p1_ylim': None, 'q2p2_xlim': None, 'q2p2_ylim': None, 'equal_aspect': True, 'legend_fontsize': 7, 'legend_markerscale': 4},
        'energy': {'figsize': (9, 4), 'ylim_rel_range': 1.3, 'force_ylim': (0.95,1.05), 'legend_fontsize': 'small'},
        'poincare': {'figsize': (7, 7), 'xlim': None, 'ylim': None, 'equal_aspect': True, 'legend_fontsize': 'small', 'legend_markerscale': 3},
        'rmse': {'figsize': (9, 4), 'ylim': None, 'log_scale': True, 'legend_fontsize': 'small'},
        'loss_total': {'figsize': (9, 5), 'ylim': None, 'log_scale': True, 'legend_fontsize': 'small'},
        'loss_detail': {'figsize': (10, 5), 'ylim': None, 'log_scale': True, 'legend_fontsize': 'small', 'legend_loc': 'upper left', 'legend_bbox': (1.02, 1.0)},
        'loss_heatmap': {'figsize': (7, 5), 'cmap': 'viridis'},
        '3d_phase': {'figsize': (8, 8), 'skip_ratio': 200, 'view_elev': 25., 'view_azim': -55., 'legend_fontsize': 'x-small', 'start_marker_size': 60}
    }

    # --- Generate All Plots ---
    base_filename = f"HH_{dynamics_label}_T{int(t_span_end)}_Neval{num_points_eval}_Ntrn{num_points_train}_Ep{epochs}"
    generate_all_plots(all_predictions, exact_data_highres, initial_state_np, initial_energy_check, dynamics_label, t_span_end, base_filename, plot_config, activations_to_test)

    # --- Save Results to Excel ---
    save_results_to_excel(all_predictions, base_filename)

    # --- Display Plots ---
    print("Displaying plots...")
    
    try:
        # Check if get_ipython is defined 
        ipython_shell = get_ipython()
        if ipython_shell is not None and ipython_shell.__class__.__name__ == 'ZMQInteractiveShell':
             
             plt.show() 
        else:
            
             if ipython_shell is not None:
               
                 if plt.isinteractive():
                     plt.show(block=True) 
                 else:
                    
                    try:
                        plt.show()
                    except Exception as e:
                        print(f"Could not display plots interactively ({e}). Plots saved to files.")

             else:
                 
                 print("Non-interactive environment detected or get_ipython not found. Plots saved to files.")
                 # plt.show() might open windows but not keep them open without blocking
                 # Or it might fail depending on the backend
                 # Consider adding plt.show(block=True) if you want the script to pause
                 # plt.show(block=True) # Uncomment if you want script to pause here
    except NameError:
        # get_ipython is not defined, definitely not in an IPython environment
        print("Standard Python environment detected. Plots saved to files.")
        # Consider adding plt.show(block=True) if you want the script to pause
        # plt.show(block=True) # Uncomment if you want script to pause here


    print("\n--- Hénon-Heiles Simulation, Plotting, and Export Complete ---")
