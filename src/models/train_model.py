import torch
import torch.nn as nn
import numpy as np

class PINN3DTemperature(nn.Module):
    def __init__(self, hidden_layers=[50, 50, 50]):
        super(PINN3DTemperature, self).__init__()
        layers = []
        input_dim = 4  # x, y, z, power
        
        # 构建隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
            
        # 输出层 (温度场)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.thermal_conductivity = nn.Parameter(torch.tensor([1.0]))
    
    def forward(self, x, y, z, power):
        inputs = torch.cat([x, y, z, power], dim=1)
        temperature = self.network(inputs)
        return temperature
    
    def compute_pde_residual(self, x, y, z, power):
        # 启用自动求导以计算偏导数
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        
        temperature = self.forward(x, y, z, power)
        
        # 计算二阶偏导数
        d_t_x = torch.autograd.grad(temperature.sum(), x, create_graph=True)[0]
        d_t_y = torch.autograd.grad(temperature.sum(), y, create_graph=True)[0]
        d_t_z = torch.autograd.grad(temperature.sum(), z, create_graph=True)[0]
        
        d2_t_x2 = torch.autograd.grad(d_t_x.sum(), x, create_graph=True)[0]
        d2_t_y2 = torch.autograd.grad(d_t_y.sum(), y, create_graph=True)[0]
        d2_t_z2 = torch.autograd.grad(d_t_z.sum(), z, create_graph=True)[0]
        
        # 热传导方程残差
        k = self.thermal_conductivity
        residual = k * (d2_t_x2 + d2_t_y2 + d2_t_z2) - power
        
        return residual

def train_pinn(model, x_train, y_train, epochs=1000, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 提取训练数据
        x = x_train['coords'][:, 0:1]
        y = x_train['coords'][:, 1:2]
        z = x_train['coords'][:, 2:3]
        power = x_train['power']
        
        # 计算物理残差
        pde_residual = model.compute_pde_residual(x, y, z, power)
        
        # 计算预测温度
        pred_temp = model(x, y, z, power)
        
        # 计算数据损失和物理损失
        data_loss = mse_loss(pred_temp, y_train['temperature'])
        physics_loss = torch.mean(torch.square(pde_residual))
        
        # 总损失
        total_loss = data_loss + physics_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.4f}, '
                  f'Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}')