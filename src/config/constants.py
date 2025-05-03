import torch

# Physical Constants
L = 1.0  # Cube side length (m)
T0_K = 293.15 # Initial temperature (K)
SIGMA = 5.67e-8 # Stefan-Boltzmann constant (W/m^2 K^4)

# Material Properties (Example: Aluminum-like)
ALPHA = 1.0e-5  # Thermal diffusivity (m^2/s)
K_VAL = 200.0   # Thermal conductivity (W/m K)
EPSILON = 0.8   # Surface emissivity

# Simulation Parameters
T_MAX = 1000.0 # Max simulation time (s)
T_SURR_MIN = 293.15 # Min surrounding temperature (K)
T_SURR_MAX = 1000.0 # Max surrounding temperature (K)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")