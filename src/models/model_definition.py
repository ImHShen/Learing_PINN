import torch
import torch.nn as nn

class PINN_Parametric(nn.Module):
    """
    Parametric Physics-Informed Neural Network model.
    Input: (x, y, z, t, T_surr)
    Output: u (Temperature)
    """
    def __init__(self, layers=[5] + [128]*6 + [1], activation=nn.Tanh()):
        super().__init__()
        if not isinstance(layers, list) or len(layers) < 2:
            raise ValueError("Layers must be a list of at least two integers.")

        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i+1])
            # Apply Xavier initialization to hidden layers' weights for better convergence
            if i < len(layers) - 2:
                 nn.init.xavier_uniform_(linear_layer.weight)
                 nn.init.zeros_(linear_layer.bias)
            # Else (last layer), use default initialization or specific if needed
            self.layers.append(linear_layer)


    def forward(self, x, y, z, t, T_surr):
        # Ensure inputs are 2D tensors (N, 1) before concatenating
        inputs = torch.cat([
            x.view(-1, 1),
            y.view(-1, 1),
            z.view(-1, 1),
            t.view(-1, 1),
            T_surr.view(-1, 1)
        ], dim=1)

        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
        # Final layer (no activation)
        output = self.layers[-1](inputs)
        return output