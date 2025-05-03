import torch
import numpy as np
from src.models.model_definition import PINN_Parametric
from config.constants import DEVICE, L, T_MAX

def predict(model, x, y, z, t, T_surr):
    """Make predictions using the trained model."""
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Prepare input tensors
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        z_t = torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        t_t = torch.tensor(t, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        T_s_t = torch.tensor(T_surr, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        # Ensure tensors require grad is False for prediction if needed
        x_t.requires_grad_(False); y_t.requires_grad_(False); z_t.requires_grad_(False)
        t_t.requires_grad_(False); T_s_t.requires_grad_(False)

        u_pred = model(x_t, y_t, z_t, t_t, T_s_t)
    return u_pred.cpu().numpy()

def create_evaluation_grid(nx=20, ny=20, nz=20, nt=10, T_surr_eval=500.0):
    """Creates a grid of points for evaluation."""
    x_lin = np.linspace(0, L, nx)
    y_lin = np.linspace(0, L, ny)
    z_lin = np.linspace(0, L, nz)
    t_lin = np.linspace(0, T_MAX, nt)

    x_grid, y_grid, z_grid, t_grid = np.meshgrid(x_lin, y_lin, z_lin, t_lin, indexing='ij')

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    t_flat = t_grid.flatten()
    T_s_flat = np.full_like(x_flat, T_surr_eval)

    return x_flat, y_flat, z_flat, t_flat, T_s_flat, (nx, ny, nz, nt)


# Example usage (typically called from main.py or notebooks)
if __name__ == '__main__':
    # Load trained model
    model_path = '../../models/trained_pinn_test.pth' # Adjust path
    layers = [5] + [128]*6 + [1] # Must match the trained model architecture
    model = PINN_Parametric(layers).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit()

    # Create grid for a specific T_surr
    T_surr_to_evaluate = 700.0 # Example evaluation T_surr
    x_eval, y_eval, z_eval, t_eval, T_s_eval, grid_shape = create_evaluation_grid(
        nx=10, ny=10, nz=10, nt=5, T_surr_eval=T_surr_to_evaluate
    )

    # Make predictions
    u_predictions = predict(model, x_eval, y_eval, z_eval, t_eval, T_s_eval)
    u_pred_grid = u_predictions.reshape(grid_shape)

    print(f"Generated predictions for T_surr = {T_surr_to_evaluate} K")
    print(f"Prediction grid shape: {u_pred_grid.shape}")
    # Next step would be to visualize these predictions (e.g., using visualize.py)
    # Example: Print temperature at center at final time
    center_idx = (grid_shape[0]//2, grid_shape[1]//2, grid_shape[2]//2)
    final_time_idx = -1
    print(f"Predicted temp at center ({L/2:.2f},{L/2:.2f},{L/2:.2f}) "
          f"at t={T_MAX:.1f}s: {u_pred_grid[center_idx][final_time_idx]:.2f} K")