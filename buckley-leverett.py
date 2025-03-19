import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Define domain parameters
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 0.5
N_collocation = 1200
N_boundary = 80
N_initial = 160

# Multiphase parameters
class MultiphaseParams:
    def __init__(self, trainable=False):
        # End-point relative permeability values
        self.k0rg = nn.Parameter(torch.tensor(0.7), requires_grad=trainable)
        self.k0rw = nn.Parameter(torch.tensor(1.0), requires_grad=trainable)
        
        # Relative permeability exponents
        self.ng = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.nw = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        
        # Residual saturations
        self.Sgr = nn.Parameter(torch.tensor(0.0), requires_grad=trainable)
        self.Swr = nn.Parameter(torch.tensor(0.2), requires_grad=trainable)
        
        # Viscosities
        self.mu_g = nn.Parameter(torch.tensor(0.02), requires_grad=False)  # or 0.2 for large mobility ratio
        self.mu_w = nn.Parameter(torch.tensor(1.0), requires_grad=False)

# Neural network architecture
class PINNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
        
    def forward(self, x):
        return self.activation(self.layer(x))

class PINN(nn.Module):
    def __init__(self, hidden_layers=6, hidden_nodes=20, activation=nn.Tanh(), trainable_params=False):
        super().__init__()
        self.activation = activation
        self.params = MultiphaseParams(trainable=trainable_params)
        
        # Neural network layers
        self.input_layer = PINNBlock(2, hidden_nodes, activation)  # Input: (x, t)
        self.hidden_layers = nn.Sequential(*[
            PINNBlock(hidden_nodes, hidden_nodes, activation)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_nodes, 1)  # Output: Sw (water saturation)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):
        """x is a tensor with shape (batch_size, 2) containing (x_D, t_D) pairs"""
        y = self.input_layer(x)
        y = self.hidden_layers(y)
        sw = self.output_layer(y)
        # Ensure saturation is bounded between Swr and 1
        return torch.sigmoid(sw) * (1 - self.params.Swr) + self.params.Swr
    
    def relative_permeability(self, sw):
        # Calculate normalized water saturation
        sw_norm = (sw - self.params.Swr) / (1 - self.params.Sgr - self.params.Swr)
        sw_norm = torch.clip(sw_norm, 0.0, 1.0)
        
        # Corey-type relative permeability functions
        krw = self.params.k0rw * (sw_norm ** self.params.nw)
        sg_norm = 1.0 - sw_norm
        krg = self.params.k0rg * (sg_norm ** self.params.ng)
        
        return krw, krg
    
    def fractional_flow(self, sw):
        """Calculate gas fractional flow function"""
        krw, krg = self.relative_permeability(sw)
        
        # Calculate mobility ratio
        mobility_ratio = (krw * self.params.mu_g) / (krg * self.params.mu_w)
        
        # Calculate gas fractional flow
        fg = 1.0 / (1.0 + mobility_ratio)
        
        return fg
    
    def dfg_dsw(self, sw):
        """Calculate derivative of gas fractional flow function w.r.t water saturation"""
        sw_detached = sw.detach().requires_grad_(True)
        fg = self.fractional_flow(sw_detached)
        dfg = torch.autograd.grad(
            fg, sw_detached, 
            grad_outputs=torch.ones_like(fg),
            create_graph=True
        )[0]
        return dfg

# Learning rate scheduler
class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        patience=16,
        cooldown_period=160,
        min_lr=1e-6,
        factor=0.8,
        eps=1e-8
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.cooldown_period = cooldown_period
        self.cooldown = cooldown_period
        self.min_lr = min_lr
        self.factor = factor
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.eps = eps
        super().__init__(optimizer)

    def step(self, curr_loss=float("inf")):
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            self.bad_epochs = 0
            return
        self.bad_epochs += 1

        # once lr is changed, further change not allowed till cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        if self.bad_epochs > self.patience:
            self.bad_epochs = 0
            for i, param_group in enumerate(self.optimizer.param_groups):
                prev_lr = param_group["lr"]
                new_lr = max(self.factor * prev_lr, self.min_lr)
                # change lr only for significant change
                if prev_lr - new_lr > self.eps * prev_lr:
                    param_group["lr"] = new_lr
                    self.cooldown = self.cooldown_period

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

# Analytical solution for comparison
def analytical_solution(x_vals, t_vals, params):
    """
    Compute the analytical solution using the Buckley-Leverett equation
    Returns water saturation profiles for given x and t values
    """
    # Simplified implementation - would need to be expanded for a real comparison
    sw_profiles = []
    
    for t in t_vals:
        # For each time, calculate water saturation profile
        sw_profile = []
        for x in x_vals:
            if x <= t * 0.5:  # Simplified shock velocity
                sw = params.Swr
            else:
                sw = 1.0  # Initial condition
            sw_profile.append(sw)
        sw_profiles.append(sw_profile)
    
    return sw_profiles

# Generate training points
def generate_training_points():
    # Collocation points (randomly distributed in domain)
    x_collocation = torch.rand(N_collocation, 1) * (x_max - x_min) + x_min
    t_collocation = torch.rand(N_collocation, 1) * (t_max - t_min) + t_min
    collocation_points = torch.cat([x_collocation, t_collocation], dim=1).to(DEVICE)
    
    # Boundary points (x=0, all t)
    t_boundary = torch.linspace(t_min, t_max, N_boundary).reshape(-1, 1)
    x_boundary = torch.zeros_like(t_boundary)
    boundary_points = torch.cat([x_boundary, t_boundary], dim=1).to(DEVICE)
    
    # Initial points (t=0, all x)
    x_initial = torch.linspace(x_min, x_max, N_initial).reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    initial_points = torch.cat([x_initial, t_initial], dim=1).to(DEVICE)
    
    return collocation_points, boundary_points, initial_points

def compute_residual(model, x_t):
    """Compute the PDE residual at collocation points"""
    x, t = x_t[:, 0:1], x_t[:, 1:2]
    
    # We need gradients w.r.t inputs
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Get water saturation prediction
    inputs = torch.cat([x, t], dim=1)
    sw = model(inputs)
    
    # Compute derivatives using autograd
    dsw_dt = torch.autograd.grad(
        sw, t, 
        grad_outputs=torch.ones_like(sw),
        create_graph=True
    )[0]
    
    dsw_dx = torch.autograd.grad(
        sw, x, 
        grad_outputs=torch.ones_like(sw),
        create_graph=True
    )[0]
    
    # Compute gas saturation
    sg = 1.0 - sw
    
    # Compute dfg/dsw (derivative of fractional flow function)
    dfg_dsw = model.dfg_dsw(sw)
    
    # Compute the PDE residual: dsw/dt + dfg/dsw * dsw/dx = 0
    # Note: dsw/dt = -dsg/dt and dfg/dsw = -dfw/dsw
    residual = dsw_dt - dfg_dsw * dsw_dx
    
    # Option to add diffusion term as in the paper
    # diffusion_term = lambda * d²sw/dx²
    # Second derivative would be computed using autograd grad twice
    
    return residual

def train_model(model, with_observed_data=False, with_diffusion=False, 
                mobility_ratio="small", num_epochs=50000):
    """Train the PINN model"""
    # Set gas viscosity based on mobility ratio
    if mobility_ratio == "large":
        model.params.mu_g.data = torch.tensor(0.2)
    else:  # small
        model.params.mu_g.data = torch.tensor(0.02)
    
    # Generate training points
    collocation_points, boundary_points, initial_points = generate_training_points()
    
    # Generate observed data (if using)
    observed_data = None
    if with_observed_data:
        # This would be replaced with actual observed data
        # For now, we'll generate synthetic data from a simplified analytical solution
        x_obs = torch.linspace(x_min, x_max, 100).to(DEVICE)
        t_obs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).to(DEVICE)
        
        # Create grid of observed points
        x_obs_grid = x_obs.repeat(len(t_obs))
        t_obs_grid = torch.repeat_interleave(t_obs, len(x_obs))
        observed_points = torch.stack([x_obs_grid, t_obs_grid], dim=1)
        
        # Generate "observed" values (simplified for this example)
        params = MultiphaseParams(trainable=False)
        if mobility_ratio == "large":
            params.mu_g = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        
        sw_obs = []
        for t_val in t_obs:
            for x_val in x_obs:
                if x_val <= 0.5 * t_val:  # Simplified shock wave
                    sw_obs.append(params.Swr.item())
                else:
                    sw_obs.append(1.0)
        
        observed_values = torch.tensor(sw_obs, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        observed_data = (observed_points, observed_values)
    
    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = LRScheduler(optimizer)
    
    # Training loop
    running_loss = []
    
    with tqdm(total=num_epochs, unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute PDE residual loss at collocation points
            residual = compute_residual(model, collocation_points)
            residual_loss = torch.mean(residual**2)
            
            # Compute initial condition loss
            sw_initial = model(initial_points)
            initial_loss = torch.mean((sw_initial - 1.0)**2)  # Initial condition: Sw = 1.0 everywhere
            
            # Compute boundary condition loss
            sw_boundary = model(boundary_points)
            boundary_loss = torch.mean((sw_boundary - model.params.Swr)**2)  # Boundary condition: Sw = Swr at x=0
            
            # Combine losses
            loss = residual_loss + initial_loss + boundary_loss
            
            # Add observed data loss if provided
            if with_observed_data:
                obs_points, obs_values = observed_data
                sw_obs_pred = model(obs_points)
                observed_loss = torch.mean((sw_obs_pred - obs_values)**2)
                loss += observed_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            
            # Record loss
            running_loss.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "lr": f"{scheduler.get_lr():.3e}"
            })
            pbar.update(1)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(running_loss)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'figures/loss_history_{mobility_ratio}_mr_diffusion={with_diffusion}_with_observed_data={with_observed_data}.png')
    
    return running_loss

def evaluate_model(model, plot_times, with_observed_data, with_diffusion, mobility_ratio="small"):
    """Evaluate and visualize the model results"""
    # Create grid for evaluation
    x_eval = torch.linspace(x_min, x_max, 100).to(DEVICE)
    
    plt.figure(figsize=(12, 8))
    
    for t_val in plot_times:
        # Create input points at time t_val
        t_eval = torch.ones_like(x_eval) * t_val
        x_t_eval = torch.stack([x_eval, t_eval], dim=1)
        
        # Get model predictions
        with torch.no_grad():
            sw_pred = model(x_t_eval).cpu().numpy()
        
        # Plot results
        plt.plot(x_eval.cpu().numpy(), sw_pred, label=f't = {t_val:.2f}')
    
    plt.xlabel('Dimensionless Distance (x_D)')
    plt.ylabel('Water Saturation (Sw)')
    plt.title(f'Water Saturation Profiles - {mobility_ratio.capitalize()} Mobility Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/saturation_profiles_{mobility_ratio}_mr_diffusion={with_diffusion}_with_observed_data={with_observed_data}.png')
    plt.show()

def main():
    # Case 1: No observed data, non-trainable parameters, no diffusion
    print("Case 1: No observed data, non-trainable parameters, no diffusion")
    model_case1_small = PINN(trainable_params=False).to(DEVICE)
    loss_case1_small = train_model(model_case1_small, with_observed_data=False, 
                           with_diffusion=False, mobility_ratio="small", num_epochs=50000)
    evaluate_model(model_case1_small, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="small", with_observed_data=False, with_diffusion=False)
    
    model_case1_large = PINN(trainable_params=False).to(DEVICE)
    loss_case1_large = train_model(model_case1_large, with_observed_data=False, 
                           with_diffusion=False, mobility_ratio="large", num_epochs=50000)
    evaluate_model(model_case1_large, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="large", with_observed_data=False, with_diffusion=False)
    
    # Case 2: With observed data, non-trainable parameters, no diffusion
    print("Case 2: With observed data, non-trainable parameters, no diffusion")
    model_case2_small = PINN(trainable_params=False).to(DEVICE)
    loss_case2_small = train_model(model_case2_small, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="small", num_epochs=50000)
    evaluate_model(model_case2_small, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="small", with_observed_data=False, with_diffusion=False)
    
    model_case2_large = PINN(trainable_params=False).to(DEVICE)
    loss_case2_large = train_model(model_case2_large, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="large", num_epochs=50000)
    evaluate_model(model_case2_large, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="large", with_observed_data=False, with_diffusion=False)
    
    # Case 3: With observed data, trainable parameters, no diffusion
    print("Case 3: With observed data, trainable parameters, no diffusion")
    model_case3_small = PINN(trainable_params=True).to(DEVICE)
    loss_case3_small = train_model(model_case3_small, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="small", num_epochs=50000)
    evaluate_model(model_case3_small, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="small", with_observed_data=False, with_diffusion=False)
    
    model_case3_large = PINN(trainable_params=True).to(DEVICE)
    loss_case3_large = train_model(model_case3_large, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="large", num_epochs=50000)
    evaluate_model(model_case3_large, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="large", with_observed_data=False, with_diffusion=False)
    
    # Case 4: With observed data, trainable parameters, with diffusion
    print("Case 4: With observed data, trainable parameters, with diffusion")
    model_case4_small = PINN(trainable_params=True).to(DEVICE)
    loss_case4_small = train_model(model_case4_small, with_observed_data=True, 
                           with_diffusion=True, mobility_ratio="small", num_epochs=50000)
    evaluate_model(model_case4_small, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="small", with_observed_data=False, with_diffusion=False)
    
    model_case4_large = PINN(trainable_params=True).to(DEVICE)
    loss_case4_large = train_model(model_case4_large, with_observed_data=True, 
                           with_diffusion=True, mobility_ratio="large", num_epochs=50000)
    evaluate_model(model_case4_large, plot_times=[0.1, 0.2, 0.3, 0.4], mobility_ratio="large", with_observed_data=False, with_diffusion=False)
    
    # Compare trained parameters if needed
    if model_case3_small.params.k0rg.requires_grad:
        print("Trained parameters for Case 3 (small mobility ratio):")
        print(f"k0rg: {model_case3_small.params.k0rg.item()}")
        print(f"k0rw: {model_case3_small.params.k0rw.item()}")
        print(f"Swr: {model_case3_small.params.Swr.item()}")
        print(f"Sgr: {model_case3_small.params.Sgr.item()}")
    
    if model_case4_large.params.k0rg.requires_grad:
        print("Trained parameters for Case 4 (large mobility ratio):")
        print(f"k0rg: {model_case4_large.params.k0rg.item()}")
        print(f"k0rw: {model_case4_large.params.k0rw.item()}")
        print(f"Swr: {model_case4_large.params.Swr.item()}")
        print(f"Sgr: {model_case4_large.params.Sgr.item()}")

if __name__ == "__main__":
    main()