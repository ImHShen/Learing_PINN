import torch
import torch.optim as optim
import time
import os
from src.models.model_definition import PINN_Parametric
from src.utils.helper_functions import sample_pde, sample_ic, sample_bc, u_initial_distribution
from config.constants import DEVICE, ALPHA, K_VAL, EPSILON, SIGMA, T0_K

def compute_loss(model, points_pde, points_ic, points_bc,
                 alpha_val, k_val, epsilon_val, sigma_val, T0_val):
    """Computes the total PINN loss."""
    # Unpack points
    x_pde, y_pde, z_pde, t_pde, T_s_pde = points_pde
    x_ic, y_ic, z_ic, t_ic, T_s_ic = points_ic
    x_bc, y_bc, z_bc, t_bc, T_s_bc, nx_bc, ny_bc, nz_bc = points_bc

    # --- PDE Loss ---
    u_pde = model(x_pde, y_pde, z_pde, t_pde, T_s_pde)
    grad_u = torch.autograd.grad(u_pde, (x_pde, y_pde, z_pde, t_pde), grad_outputs=torch.ones_like(u_pde), create_graph=True)
    u_t, u_x, u_y, u_z = grad_u[3], grad_u[0], grad_u[1], grad_u[2]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_pde, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z_pde, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    laplacian_u = u_xx + u_yy + u_zz
    pde_residual = u_t - alpha_val * laplacian_u
    loss_pde = torch.mean(pde_residual**2)

    # --- IC Loss ---
    u_ic_pred = model(x_ic, y_ic, z_ic, t_ic, T_s_ic)
    u_ic_true = u_initial_distribution(x_ic, y_ic, z_ic) # Get true IC value
    ic_residual = u_ic_pred - u_ic_true
    loss_ic = torch.mean(ic_residual**2)

    # --- BC Loss ---
    u_bc = model(x_bc, y_bc, z_bc, t_bc, T_s_bc)
    u_bc_phys = torch.relu(u_bc) + 1e-6 # Ensure temp >= 0 K approx.
    T_s_bc_phys = torch.relu(T_s_bc) + 1e-6

    grad_u_bc = torch.autograd.grad(u_bc, (x_bc, y_bc, z_bc), grad_outputs=torch.ones_like(u_bc), create_graph=True)
    grad_u_dot_n = grad_u_bc[0]*nx_bc + grad_u_bc[1]*ny_bc + grad_u_bc[2]*nz_bc
    bc_residual = -k_val * grad_u_dot_n - epsilon_val * sigma_val * (T_s_bc_phys**4 - u_bc_phys**4)
    loss_bc = torch.mean(bc_residual**2)

    # --- Total Loss (Weights need tuning) ---
    lambda_pde = 1.0
    lambda_ic = 10.0
    lambda_bc = 5.0
    total_loss = (lambda_pde * loss_pde +
                  lambda_ic * loss_ic +
                  lambda_bc * loss_bc)

    return total_loss, loss_pde, loss_ic, loss_bc


def train(config):
    """Main training function."""
    print(f"Using device: {DEVICE}")

    # Hyperparameters from config
    layers = config.get('layers', [5] + [128]*6 + [1])
    learning_rate = config.get('learning_rate', 1e-4)
    epochs = config.get('epochs', 50000)
    n_pde = config.get('n_pde', 4096)
    n_ic = config.get('n_ic', 1024)
    n_bc_face = config.get('n_bc_face', 512)
    model_save_path = config.get('model_save_path', 'models/trained_pinn.pth')
    log_frequency = config.get('log_frequency', 500)

    # Initialize model and optimizer
    model = PINN_Parametric(layers).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.7)

    # Constants to device
    alpha_d = torch.tensor(ALPHA, dtype=torch.float32, device=DEVICE)
    k_val_d = torch.tensor(K_VAL, dtype=torch.float32, device=DEVICE)
    epsilon_d = torch.tensor(EPSILON, dtype=torch.float32, device=DEVICE)
    sigma_d = torch.tensor(SIGMA, dtype=torch.float32, device=DEVICE)
    T0_d = torch.tensor(T0_K, dtype=torch.float32, device=DEVICE)

    print("Starting training...")
    start_time = time.time()

    # Training history (optional)
    history = {'loss': [], 'loss_pde': [], 'loss_ic': [], 'loss_bc': []}

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample points
        pde_points = sample_pde(n_pde)
        ic_points = sample_ic(n_ic)
        bc_points = sample_bc(n_bc_face)

        # Calculate loss
        loss, lp, li, lb = compute_loss(
            model, pde_points, ic_points, bc_points,
            alpha_d, k_val_d, epsilon_d, sigma_d, T0_d
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log history
        history['loss'].append(loss.item())
        history['loss_pde'].append(lp.item())
        history['loss_ic'].append(li.item())
        history['loss_bc'].append(lb.item())


        # Print progress
        if (epoch + 1) % log_frequency == 0:
            elapsed = time.time() - start_time
            lr_current = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4e}, '
                  f'PDE: {lp.item():.3e}, IC: {li.item():.3e}, BC: {lb.item():.3e}, '
                  f'LR: {lr_current:.2e}, Time: {elapsed:.2f}s')
            start_time = time.time() # Reset timer for interval timing

    print("Training finished.")

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Return training history (optional)
    return history

# Example usage (typically called from main.py)
if __name__ == '__main__':
    # Example configuration
    training_config = {
        'layers': [5] + [128]*6 + [1],
        'learning_rate': 1e-4,
        'epochs': 1000, # Use a smaller number for quick test
        'n_pde': 1024,
        'n_ic': 256,
        'n_bc_face': 128,
        'model_save_path': '../../models/trained_pinn_test.pth', # Adjust path relative to script
        'log_frequency': 100
    }
    train(training_config)