import torch
import torch.optim as optim
import time
import os
import numpy as np # Usually needed

# Use absolute imports from src
from src.models.model_definition import PINN_Parametric
from src.utils.helper_functions import sample_pde, sample_ic, sample_bc, u_initial_distribution
from src.utils.pytorch_utils import DEVICE # Import device setting


def compute_loss(model, points_pde, points_ic, points_bc, params):
    """Computes the total PINN loss using parameters from dict."""
    # Unpack points - ensure they are tensors on the correct device already
    x_pde, y_pde, z_pde, t_pde, T_s_pde = points_pde
    x_ic, y_ic, z_ic, t_ic, T_s_ic = points_ic
    x_bc, y_bc, z_bc, t_bc, T_s_bc, nx_bc, ny_bc, nz_bc = points_bc

    # Unpack parameters needed for loss calculation (already tensors on device)
    alpha_val = params['alpha']
    k_val = params['k']
    epsilon_val = params['epsilon']
    sigma_val = params['sigma']
    T0_val = params['T0_K'] # Initial temp value

    # --- PDE Loss ---
    u_pde = model(x_pde, y_pde, z_pde, t_pde, T_s_pde)
    # Compute derivatives using torch.autograd.grad
    # Ensure requires_grad=True was set during sampling for coordinate inputs
    grad_u = torch.autograd.grad(
        outputs=u_pde,
        inputs=(x_pde, y_pde, z_pde, t_pde),
        grad_outputs=torch.ones_like(u_pde),
        create_graph=True # Keep graph for higher derivatives
    )
    u_t, u_x, u_y, u_z = grad_u[3], grad_u[0], grad_u[1], grad_u[2]

    # Compute second derivatives (Laplacian) - requires create_graph=True above
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_pde, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z_pde, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    laplacian_u = u_xx + u_yy + u_zz

    pde_residual = u_t - alpha_val * laplacian_u
    loss_pde = torch.mean(pde_residual**2)

    # --- IC Loss ---
    u_ic_pred = model(x_ic, y_ic, z_ic, t_ic, T_s_ic)
    # u_ic_true needs T0_val now
    u_ic_true = u_initial_distribution(x_ic, y_ic, z_ic, T0_val)
    ic_residual = u_ic_pred - u_ic_true
    loss_ic = torch.mean(ic_residual**2)

    # --- BC Loss ---
    u_bc = model(x_bc, y_bc, z_bc, t_bc, T_s_bc)
    # Ensure predicted temperature doesn't go below absolute zero (important for u^4)
    u_bc_phys = torch.relu(u_bc) + 1e-8 # Small epsilon for stability

    # Compute normal derivative (requires gradients w.r.t spatial coords)
    grad_u_bc = torch.autograd.grad(
        outputs=u_bc,
        inputs=(x_bc, y_bc, z_bc),
        grad_outputs=torch.ones_like(u_bc),
        create_graph=True # May not be needed if no higher derivatives of BC loss
    )
    u_x_bc, u_y_bc, u_z_bc = grad_u_bc[0], grad_u_bc[1], grad_u_bc[2]
    grad_u_dot_n = u_x_bc * nx_bc + u_y_bc * ny_bc + u_z_bc * nz_bc

    # Ensure T_surr is also non-negative (should be by sampling)
    T_s_bc_phys = torch.relu(T_s_bc) + 1e-8

    # BC Residual: -k * du/dn - epsilon * sigma * (T_surr^4 - u^4) = 0
    bc_residual = -k_val * grad_u_dot_n - epsilon_val * sigma_val * (T_s_bc_phys**4 - u_bc_phys**4)
    loss_bc = torch.mean(bc_residual**2)

    # --- Total Loss ---
    # Get weights from params dict, provide defaults
    lambda_pde = params.get('lambda_pde', 1.0)
    lambda_ic = params.get('lambda_ic', 10.0)
    lambda_bc = params.get('lambda_bc', 5.0)

    total_loss = (lambda_pde * loss_pde +
                  lambda_ic * loss_ic +
                  lambda_bc * loss_bc)

    # Return individual losses for logging
    return total_loss, loss_pde, loss_ic, loss_bc


def train(config):
    """Main training function, using config dictionary."""
    print(f"Using device: {DEVICE}") # Using imported DEVICE

    # --- Extract parameters from config dictionary ---
    problem_cfg = config.get('problem', {})
    material_cfg = config.get('material', {})
    const_cfg = config.get('constants', {})
    train_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})

    # Problem / Simulation parameters
    L_val = problem_cfg.get('L', 1.0)
    T0_val = problem_cfg.get('T0_K', 293.15)
    T_max_val = problem_cfg.get('T_max', 1000.0)
    T_surr_min_val = problem_cfg.get('T_surr_min', 293.15)
    T_surr_max_val = problem_cfg.get('T_surr_max', 1000.0)

    # Material / Physics parameters (convert to tensors on device)
    loss_params = {
        'alpha': torch.tensor(material_cfg.get('alpha', 1e-5), dtype=torch.float32, device=DEVICE),
        'k': torch.tensor(material_cfg.get('k', 200.0), dtype=torch.float32, device=DEVICE),
        'epsilon': torch.tensor(material_cfg.get('epsilon', 0.8), dtype=torch.float32, device=DEVICE),
        'sigma': torch.tensor(const_cfg.get('sigma', 5.67e-8), dtype=torch.float32, device=DEVICE),
        'T0_K': torch.tensor(T0_val, dtype=torch.float32, device=DEVICE),
        # Add loss weights from config if specified, else use defaults in compute_loss
        'lambda_pde': train_cfg.get('lambda_pde', 1.0),
        'lambda_ic': train_cfg.get('lambda_ic', 10.0),
        'lambda_bc': train_cfg.get('lambda_bc', 5.0)
    }

    # Training Hyperparameters
    layers = train_cfg.get('layers', [5] + [128]*6 + [1])
    learning_rate = train_cfg.get('learning_rate', 1e-4)
    epochs = train_cfg.get('epochs', 50000)
    batch_sizes = train_cfg.get('batch_sizes', {'pde': 4096, 'ic': 1024, 'bc_face': 512})
    model_save_path = paths_cfg.get('model_save', 'models/trained_pinn.pth') # Path relative to root
    log_frequency = train_cfg.get('log_frequency', 500)

    # --- Initialize model and optimizer ---
    model = PINN_Parametric(layers).to(DEVICE)
    # Consider different optimizers based on config (e.g., AdamW, LBFGS)
    optimizer_name = train_cfg.get('optimizer', 'Adam').lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Add other optimizers like LBFGS if needed (requires closure function)
    # elif optimizer_name == 'lbfgs':
    #     optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=...)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}'. Using Adam.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                         step_size=train_cfg.get('scheduler_step_size', 5000),
                                         gamma=train_cfg.get('scheduler_gamma', 0.7))

    print("Starting training...")
    start_time = time.time()
    history = {'loss': [], 'loss_pde': [], 'loss_ic': [], 'loss_bc': []}

    # --- Training Loop ---
    for epoch in range(epochs):

        # --- Closure for LBFGS (if used) ---
        # def closure():
        #     optimizer.zero_grad()
        #     # Sample points inside closure if using LBFGS
        #     pde_points = sample_pde(...)
        #     # ... sample others ...
        #     loss, _, _, _ = compute_loss(model, pde_points, ic_points, bc_points, loss_params)
        #     loss.backward()
        #     return loss

        # --- Standard optimization step (Adam) ---
        if optimizer_name == 'adam':
            optimizer.zero_grad()

            # Sample points for this batch
            pde_points = sample_pde(batch_sizes['pde'], L_val, T_max_val, T_surr_min_val, T_surr_max_val)
            ic_points = sample_ic(batch_sizes['ic'], L_val, T_surr_min_val, T_surr_max_val)
            bc_points = sample_bc(batch_sizes['bc_face'], L_val, T_max_val, T_surr_min_val, T_surr_max_val)

            # Calculate loss
            loss, lp, li, lb = compute_loss(
                model, pde_points, ic_points, bc_points, loss_params
            )

            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            scheduler.step() # Update learning rate

        # --- LBFGS optimization step ---
        # elif optimizer_name == 'lbfgs':
        #     # LBFGS needs points sampled inside closure or passed differently
        #     loss = optimizer.step(closure)
        #     scheduler.step() # Scheduler might need adjustment for LBFGS
        #     # Need to get individual losses separately if logging

        # Log history (ensure lp, li, lb are calculated even if using LBFGS)
        history['loss'].append(loss.item())
        history['loss_pde'].append(lp.item() if 'lp' in locals() else np.nan)
        history['loss_ic'].append(li.item() if 'li' in locals() else np.nan)
        history['loss_bc'].append(lb.item() if 'lb' in locals() else np.nan)

        # Print progress
        if (epoch + 1) % log_frequency == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr_current = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4e}, '
                  f'PDE: {lp.item():.3e}, IC: {li.item():.3e}, BC: {lb.item():.3e}, '
                  f'LR: {lr_current:.2e}, Time Since Last: {elapsed:.2f}s')
            start_time = time.time() # Reset timer

    print("Training finished.")

    # Ensure model save directory exists (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_save_abs_path = os.path.join(project_root, model_save_path)
    os.makedirs(os.path.dirname(model_save_abs_path), exist_ok=True)

    torch.save(model.state_dict(), model_save_abs_path)
    print(f"Model saved to {model_save_abs_path}")

    return history