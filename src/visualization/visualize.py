import matplotlib.pyplot as plt
import numpy as np
import os
# Optional: Import pyvista for 3D plots
# try:
#     import pyvista as pv
#     pyvista_available = True
# except ImportError:
#     pyvista_available = False
#     print("Warning: pyvista not installed. 3D plotting disabled.")

def plot_loss_history(history, save_path='reports/figures/loss_curve.png'):
    """Plots the training loss history."""
    if not history or not history.get('loss'):
        print("Warning: No loss history data provided to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Total Loss', linewidth=2)
    # Plot individual losses if available and non-empty
    if history.get('loss_pde') and any(not np.isnan(x) for x in history['loss_pde']):
         plt.plot(history['loss_pde'], label='PDE Loss', alpha=0.7)
    if history.get('loss_ic') and any(not np.isnan(x) for x in history['loss_ic']):
         plt.plot(history['loss_ic'], label='IC Loss', alpha=0.7)
    if history.get('loss_bc') and any(not np.isnan(x) for x in history['loss_bc']):
         plt.plot(history['loss_bc'], label='BC Loss', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.yscale('log')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Loss curve saved to {save_path}")


def plot_temperature_slice(u_pred_grid, grid_shape, slice_dim='z', slice_index=None, time_index=-1,
                           T_surr_eval=None, save_path='reports/figures/temp_slice.png',
                           L_val=1.0, T_max_val=1000.0):
    """Plots a 2D slice of the 3D+T temperature field."""
    if u_pred_grid is None or not isinstance(u_pred_grid, np.ndarray):
        print("Warning: Invalid prediction data for plotting.")
        return

    nx, ny, nz, nt = grid_shape

    # Validate time_index
    if time_index < 0:
        time_index = nt + time_index # Convert negative index
    if not (0 <= time_index < nt):
        print(f"Warning: Invalid time_index {time_index}. Using last time step {nt-1}.")
        time_index = nt - 1
    current_time = time_index / max(1, nt - 1) * T_max_val if nt > 1 else 0

    plt.figure(figsize=(8, 6))
    vmin = np.min(u_pred_grid) if u_pred_grid.size > 0 else 273
    vmax = np.max(u_pred_grid) if u_pred_grid.size > 0 else 1000

    if slice_dim == 'z':
        if slice_index is None: slice_index = nz // 2
        if not (0 <= slice_index < nz): slice_index = nz // 2
        data_slice = u_pred_grid[:, :, slice_index, time_index]
        x_coords = np.linspace(0, L_val, nx)
        y_coords = np.linspace(0, L_val, ny)
        plt.contourf(x_coords, y_coords, data_slice.T, levels=50, cmap='inferno', vmin=vmin, vmax=vmax)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        slice_pos = slice_index / max(1, nz - 1) * L_val if nz > 1 else 0
        title = f'Temperature Slice at Z = {slice_pos:.2f}m, '
    elif slice_dim == 'y':
        if slice_index is None: slice_index = ny // 2
        if not (0 <= slice_index < ny): slice_index = ny // 2
        data_slice = u_pred_grid[:, slice_index, :, time_index]
        x_coords = np.linspace(0, L_val, nx)
        z_coords = np.linspace(0, L_val, nz)
        plt.contourf(x_coords, z_coords, data_slice.T, levels=50, cmap='inferno', vmin=vmin, vmax=vmax)
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        slice_pos = slice_index / max(1, ny - 1) * L_val if ny > 1 else 0
        title = f'Temperature Slice at Y = {slice_pos:.2f}m, '
    elif slice_dim == 'x':
        if slice_index is None: slice_index = nx // 2
        if not (0 <= slice_index < nx): slice_index = nx // 2
        data_slice = u_pred_grid[slice_index, :, :, time_index]
        y_coords = np.linspace(0, L_val, ny)
        z_coords = np.linspace(0, L_val, nz)
        plt.contourf(y_coords, z_coords, data_slice.T, levels=50, cmap='inferno', vmin=vmin, vmax=vmax)
        plt.xlabel('Y (m)')
        plt.ylabel('Z (m)')
        slice_pos = slice_index / max(1, nx - 1) * L_val if nx > 1 else 0
        title = f'Temperature Slice at X = {slice_pos:.2f}m, '
    else:
        print(f"Error: Invalid slice_dim '{slice_dim}'. Choose 'x', 'y', or 'z'.")
        plt.close()
        return

    title += f't = {current_time:.1f}s'
    if T_surr_eval is not None:
         title += f', T_surr={T_surr_eval:.1f}K'

    plt.colorbar(label='Temperature (K)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.tight_layout()
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Temperature slice saved to {save_path}")


# Optional: Add 3D plotting function using pyvista if installed
# def plot_3d_temperature(u_pred_grid, grid_shape, time_index=-1, T_surr_eval=None, save_path='reports/figures/temp_3d.png', L_val=1.0):
#     if not pyvista_available:
#         print("Pyvista not available, skipping 3D plot.")
#         return
#     # ... (Implementation using pyvista StructuredGrid) ...
#     print(f"3D temperature plot saved to {save_path}")