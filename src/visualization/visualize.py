# src/visualization/visualize.py
import matplotlib.pyplot as plt
import numpy as np
# Consider using pyvista for 3D plots if needed

def plot_loss_history(history, save_path='reports/figures/loss_curve.png'):
    """Plots the training loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['loss_pde'], label='PDE Loss', alpha=0.7)
    plt.plot(history['loss_ic'], label='IC Loss', alpha=0.7)
    plt.plot(history['loss_bc'], label='BC Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.yscale('log')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_temperature_slice(u_pred_grid, grid_shape, slice_dim='z', slice_index=None, time_index=-1, T_surr_eval=None, save_path='reports/figures/temp_slice.png'):
    """Plots a 2D slice of the 3D+T temperature field."""
    nx, ny, nz, nt = grid_shape
    if slice_index is None:
        slice_index = nz // 2 if slice_dim == 'z' else (ny // 2 if slice_dim == 'y' else nx // 2)

    plt.figure(figsize=(8, 6))
    if slice_dim == 'z':
        data_slice = u_pred_grid[:, :, slice_index, time_index]
        x_label, y_label = 'X', 'Y'
        x_extent, y_extent = [0, L], [0, L]
        title = f'Temperature Slice at Z = {slice_index / (nz - 1) * L:.2f}, '
    elif slice_dim == 'y':
        data_slice = u_pred_grid[:, slice_index, :, time_index]
        x_label, y_label = 'X', 'Z'
        x_extent, y_extent = [0, L], [0, L]
        title = f'Temperature Slice at Y = {slice_index / (ny - 1) * L:.2f}, '
    else: # slice_dim == 'x'
        data_slice = u_pred_grid[slice_index, :, :, time_index]
        x_label, y_label = 'Y', 'Z'
        x_extent, y_extent = [0, L], [0, L]
        title = f'Temperature Slice at X = {slice_index / (nx - 1) * L:.2f}, '

    title += f't = {time_index / (nt - 1) * T_MAX:.1f}s'
    if T_surr_eval is not None:
         title += f', T_surr={T_surr_eval:.1f}K'

    im = plt.imshow(data_slice.T, origin='lower', extent=x_extent + y_extent, cmap='inferno', aspect='equal') # Transpose needed? Check orientation
    plt.colorbar(im, label='Temperature (K)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
     # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Temperature slice saved to {save_path}")

# Add other visualization functions (e.g., using pyvista for 3D)

# Example usage (typically called from main.py or notebooks)
if __name__ == '__main__':
     # Example: Load dummy prediction data and plot slice
     # This part would normally get data from evaluate_model.py
     shape = (10, 10, 10, 5)
     dummy_preds = np.random.rand(*shape) * 700 + 300 # Random temps 300-1000K
     dummy_T_surr = 700.0
     plot_temperature_slice(dummy_preds, shape, slice_dim='z', T_surr_eval=dummy_T_surr, save_path='../../reports/figures/dummy_temp_slice.png') # Adjust path