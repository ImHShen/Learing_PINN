import torch
import torch.nn as nn

class PINN_Parametric(nn.Module):
    """Parametric PINN Network."""
    def __init__(self, layers=[5] + [128]*6 + [1], activation=nn.Tanh()):
        super().__init__()
        # Input: x, y, z, t, T_surr
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2: # Apply Xavier init only to hidden layers' weights
                 nn.init.xavier_uniform_(self.layers[-1].weight)
                 nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x, y, z, t, T_surr):
        inputs = torch.cat([x, y, z, t, T_surr], dim=1)
        for i in range(len(self.layers) - 2):
            inputs = self.activation(self.layers[i](inputs))
        output = self.layers[-1](inputs) # Output: u (temperature)
        return output