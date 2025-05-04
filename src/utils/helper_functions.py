import torch
# DEVICE is imported from pytorch_utils
from .pytorch_utils import DEVICE

# Functions now need config parameters passed to them
# (No direct import of constants)

def sample_pde(N, L, T_max, T_surr_min, T_surr_max):
    """Sample points for PDE loss."""
    x = torch.rand(N, 1, device=DEVICE) * L
    y = torch.rand(N, 1, device=DEVICE) * L
    z = torch.rand(N, 1, device=DEVICE) * L
    # Sample t > 0 for PDE
    t = torch.rand(N, 1, device=DEVICE) * T_max
    # Add small epsilon to avoid t=0 if desired, but rand should make it unlikely
    # t = t.clamp(min=1e-9) # Example if strict t>0 needed

    T_s = torch.rand(N, 1, device=DEVICE) * (T_surr_max - T_surr_min) + T_surr_min
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)
    return x, y, z, t, T_s

def sample_ic(N, L, T_surr_min, T_surr_max):
    """Sample points for IC loss."""
    x = torch.rand(N, 1, device=DEVICE) * L
    y = torch.rand(N, 1, device=DEVICE) * L
    z = torch.rand(N, 1, device=DEVICE) * L
    t = torch.zeros_like(x) # t = 0 for IC
    T_s = torch.rand(N, 1, device=DEVICE) * (T_surr_max - T_surr_min) + T_surr_min
    return x, y, z, t, T_s

def sample_bc(N_per_face, L, T_max, T_surr_min, T_surr_max):
    """Sample points for BC loss across all 6 faces."""
    N = N_per_face * 6
    # Sample common time (t>0) and T_surr for all boundary points
    t_bc = torch.rand(N, 1, device=DEVICE) * T_max
    # t_bc = t_bc.clamp(min=1e-9) # Ensure t > 0 if needed
    T_s_bc = torch.rand(N, 1, device=DEVICE) * (T_surr_max - T_surr_min) + T_surr_min

    # Define faces and normals
    face_defs = [
        {'dim': 0, 'val': 0.0, 'n': (-1.0, 0.0, 0.0)}, # x=0
        {'dim': 0, 'val': L,   'n': ( 1.0, 0.0, 0.0)}, # x=L
        {'dim': 1, 'val': 0.0, 'n': (0.0, -1.0, 0.0)}, # y=0
        {'dim': 1, 'val': L,   'n': (0.0,  1.0, 0.0)}, # y=L
        {'dim': 2, 'val': 0.0, 'n': (0.0, 0.0, -1.0)}, # z=0
        {'dim': 2, 'val': L,   'n': (0.0, 0.0,  1.0)}, # z=L
    ]

    all_coords_list = []
    all_normals_list = []

    for face in face_defs:
        coords = [None, None, None]
        dims_to_sample = [d for d in range(3) if d != face['dim']]

        coords[dims_to_sample[0]] = torch.rand(N_per_face, 1, device=DEVICE) * L
        coords[dims_to_sample[1]] = torch.rand(N_per_face, 1, device=DEVICE) * L
        coords[face['dim']] = torch.full((N_per_face, 1), face['val'], device=DEVICE)

        nx = torch.full((N_per_face, 1), face['n'][0], device=DEVICE)
        ny = torch.full((N_per_face, 1), face['n'][1], device=DEVICE)
        nz = torch.full((N_per_face, 1), face['n'][2], device=DEVICE)

        # Store coordinates as (x, y, z) columns
        xyz_face = torch.cat(coords, dim=1) # Shape (N_per_face, 3)
        n_face = torch.cat([nx, ny, nz], dim=1) # Shape (N_per_face, 3)

        all_coords_list.append(xyz_face)
        all_normals_list.append(n_face)

    # Concatenate all faces
    xyz_bc = torch.cat(all_coords_list, dim=0) # Shape (N, 3)
    n_bc = torch.cat(all_normals_list, dim=0)   # Shape (N, 3)

    x_bc = xyz_bc[:, 0:1].requires_grad_(True)
    y_bc = xyz_bc[:, 1:2].requires_grad_(True)
    z_bc = xyz_bc[:, 2:3].requires_grad_(True)
    t_bc = t_bc.requires_grad_(True) # Time also needs grad for potential d(BC)/dt terms if BC were time-dependent
    nx_bc, ny_bc, nz_bc = n_bc[:, 0:1], n_bc[:, 1:2], n_bc[:, 2:3]

    return x_bc, y_bc, z_bc, t_bc, T_s_bc, nx_bc, ny_bc, nz_bc


def u_initial_distribution(x, y, z, T0_val):
    """Defines the initial temperature distribution using T0 from config."""
    # Assumes T0_val is already loaded from config and passed in
    return torch.full_like(x, T0_val)