import torch
import numpy as np
import os

# Use absolute imports from src
from src.models.model_definition import PINN_Parametric
from src.utils.pytorch_utils import DEVICE # Import device setting


def predict(model, x, y, z, t, T_surr):
    """Make predictions using the trained model."""
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for prediction
        # Prepare input tensors - ensure correct shape and device
        # Convert numpy arrays (likely input) to tensors
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE).view(-1, 1)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE).view(-1, 1)
        z_t = torch.tensor(z, dtype=torch.float32, device=DEVICE).view(-1, 1)
        t_t = torch.tensor(t, dtype=torch.float32, device=DEVICE).view(-1, 1)
        T_s_t = torch.tensor(T_surr, dtype=torch.float32, device=DEVICE).view(-1, 1)

        u_pred = model(x_t, y_t, z_t, t_t, T_s_t)
    # Return predictions as numpy array on CPU
    return u_pred.cpu().numpy()


def create_evaluation_grid(nx=20, ny=20, nz=20, nt=10, T_surr_eval=500.0, L_val=1.0, T_max_val=1000.0):
    """
    Creates a flattened grid of points (x, y, z, t, T_surr) for evaluation.
    Needs L and T_max from config.
    """
    x_lin = np.linspace(0, L_val, nx)
    y_lin = np.linspace(0, L_val, ny)
    z_lin = np.linspace(0, L_val, nz)
    t_lin = np.linspace(0, T_max_val, nt) # Include t=0 and t=T_max

    # Create meshgrid
    x_grid, y_grid, z_grid, t_grid = np.meshgrid(x_lin, y_lin, z_lin, t_lin, indexing='ij')

    # Flatten arrays
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    t_flat = t_grid.flatten()

    # Create corresponding T_surr array
    T_s_flat = np.full_like(x_flat, T_surr_eval)

    # Return flattened coordinates and the original grid shape for reshaping later
    grid_shape = (nx, ny, nz, nt)
    return x_flat, y_flat, z_flat, t_flat, T_s_flat, grid_shape

# Example standalone execution (can be removed if only called from main.py)
if __name__ == '__main__':
    # This block demonstrates direct usage but normally main.py handles workflow
    print("Running evaluate_model.py directly (for demonstration).")

    # --- Dummy Configuration (replace with loading from main.py normally) ---
    MODEL_PATH_REL = 'models/pinn_3d_rad.pth' # Example path
    LAYERS = [5] + [128]*6 + [1]
    L_VAL_DEMO = 1.0
    T_MAX_DEMO = 1000.0
    T_SURR_EVAL_DEMO = 750.0
    GRID_CONF_DEMO = {'nx': 5, 'ny': 5, 'nz': 5, 'nt': 3} # Small grid for demo
    # --- End Dummy Configuration ---

    project_root_demo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path_abs_demo = os.path.join(project_root_demo, MODEL_PATH_REL)

    # Load trained model
    model = PINN_Parametric(LAYERS).to(DEVICE)
    if os.path.exists(model_path_abs_demo):
        model.load_state_dict(torch.load(model_path_abs_demo, map_location=DEVICE))
        print(f"Model loaded from {model_path_abs_demo}")
    else:
        print(f"Warning: Model file not found at {model_path_abs_demo}. Cannot evaluate.")
        # In a real scenario, exit or handle this error
        exit()


    # Create grid for the specified T_surr
    x_eval, y_eval, z_eval, t_eval, T_s_eval, grid_shape = create_evaluation_grid(
        nx=GRID_CONF_DEMO['nx'], ny=GRID_CONF_DEMO['ny'], nz=GRID_CONF_DEMO['nz'], nt=GRID_CONF_DEMO['nt'],
        T_surr_eval=T_SURR_EVAL_DEMO, L_val=L_VAL_DEMO, T_max_val=T_MAX_DEMO
    )

    # Make predictions
    u_predictions = predict(model, x_eval, y_eval, z_eval, t_eval, T_s_eval)
    u_pred_grid = u_predictions.reshape(grid_shape)

    print(f"\nGenerated predictions for T_surr = {T_SURR_EVAL_DEMO} K")
    print(f"Prediction grid shape: {u_pred_grid.shape}")

    # Example: Print temperature at one corner (0,0,0) at final time
    corner_idx = (0, 0, 0)
    final_time_idx = -1 # Index for last time step
    t_final_val = T_MAX_DEMO
    print(f"Predicted temp at corner ({0:.2f},{0:.2f},{0:.2f}) "
          f"at t={t_final_val:.1f}s: {u_pred_grid[corner_idx][final_time_idx]:.2f} K")

    # Example: Print temperature at center at final time
    center_nx, center_ny, center_nz = grid_shape[0]//2, grid_shape[1]//2, grid_shape[2]//2
    print(f"Predicted temp at center ({L_VAL_DEMO/2:.2f},{L_VAL_DEMO/2:.2f},{L_VAL_DEMO/2:.2f}) "
          f"at t={t_final_val:.1f}s: {u_pred_grid[center_nx, center_ny, center_nz, final_time_idx]:.2f} K")