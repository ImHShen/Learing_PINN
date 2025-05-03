import torch
from ..config.constants import L, T_MAX, T_SURR_MIN, T_SURR_MAX, T0_K

def sample_pde(N):
    """Sample points for PDE loss."""
    x = torch.rand(N, 1, device=DEVICE) * L
    y = torch.rand(N, 1, device=DEVICE) * L
    z = torch.rand(N, 1, device=DEVICE) * L
    t = torch.rand(N, 1, device=DEVICE) * T_MAX
    T_s = torch.rand(N, 1, device=DEVICE) * (T_SURR_MAX - T_SURR_MIN) + T_SURR_MIN
    # Ensure points require gradient for derivative calculation
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)
    # T_s usually doesn't need grad unless inferring it
    return x, y, z, t, T_s

def sample_ic(N):
    """Sample points for Initial Condition loss."""
    x = torch.rand(N, 1, device=DEVICE) * L
    y = torch.rand(N, 1, device=DEVICE) * L
    z = torch.rand(N, 1, device=DEVICE) * L
    t = torch.zeros_like(x)
    T_s = torch.rand(N, 1, device=DEVICE) * (T_SURR_MAX - T_SURR_MIN) + T_SURR_MIN
    return x, y, z, t, T_s

def sample_bc(N_per_face):
    """Sample points for Boundary Condition loss across all 6 faces."""
    faces_coords = []
    faces_normals = []
    N = N_per_face * 6

    # Sample common time and T_surr for all boundary points in this batch
    t_bc = torch.rand(N, 1, device=DEVICE) * T_MAX
    T_s_bc = torch.rand(N, 1, device=DEVICE) * (T_SURR_MAX - T_SURR_MIN) + T_SURR_MIN

    # Define faces and normals
    face_defs = [
        {'dim': 0, 'val': 0.0, 'n': (-1.0, 0.0, 0.0)}, # x=0
        {'dim': 0, 'val': L,   'n': ( 1.0, 0.0, 0.0)}, # x=L
        {'dim': 1, 'val': 0.0, 'n': (0.0, -1.0, 0.0)}, # y=0
        {'dim': 1, 'val': L,   'n': (0.0,  1.0, 0.0)}, # y=L
        {'dim': 2, 'val': 0.0, 'n': (0.0, 0.0, -1.0)}, # z=0
        {'dim': 2, 'val': L,   'n': (0.0, 0.0,  1.0)}, # z=L
    ]

    all_coords = []
    all_normals = []

    for i, face in enumerate(face_defs):
        coords = [None, None, None]
        dims_to_sample = [d for d in range(3) if d != face['dim']]

        # Sample coordinates for the other two dimensions
        coords[dims_to_sample[0]] = torch.rand(N_per_face, 1, device=DEVICE) * L
        coords[dims_to_sample[1]] = torch.rand(N_per_face, 1, device=DEVICE) * L
        coords[face['dim']] = torch.full((N_per_face, 1), face['val'], device=DEVICE)

        # Create normal vectors
        nx = torch.full((N_per_face, 1), face['n'][0], device=DEVICE)
        ny = torch.full((N_per_face, 1), face['n'][1], device=DEVICE)
        nz = torch.full((N_per_face, 1), face['n'][2], device=DEVICE)

        all_coords.append(torch.cat(coords, dim=1)) # Shape (N_per_face, 3)
        all_normals.append(torch.cat([nx, ny, nz], dim=1)) # Shape (N_per_face, 3)

    # Concatenate all faces
    xyz_bc = torch.cat(all_coords, dim=0) # Shape (N, 3)
    n_bc = torch.cat(all_normals, dim=0)   # Shape (N, 3)

    x_bc = xyz_bc[:, 0:1].requires_grad_(True)
    y_bc = xyz_bc[:, 1:2].requires_grad_(True)
    z_bc = xyz_bc[:, 2:3].requires_grad_(True)
    t_bc = t_bc.requires_grad_(True)
    nx_bc, ny_bc, nz_bc = n_bc[:, 0:1], n_bc[:, 1:2], n_bc[:, 2:3]

    return x_bc, y_bc, z_bc, t_bc, T_s_bc, nx_bc, ny_bc, nz_bc


def u_initial_distribution(x, y, z):
    """Defines the initial temperature distribution."""
    # Uniform initial temperature
    return torch.full_like(x, T0_K)

# Add any other common helper functions here (e.g., plotting setup)