import plotly.graph_objects as go
import numpy as np
import torch

def plot_3d_temperature_field(model, domain_size=(1.0, 1.0, 1.0), power=50.0, resolution=20):
    """
    绘制3D温度场分布
    
    参数:
    model: 训练好的PINN模型
    domain_size: 空间域大小
    power: 加热功率
    resolution: 每个维度的网格点数
    """
    # 创建网格点
    x = np.linspace(0, domain_size[0], resolution)
    y = np.linspace(0, domain_size[1], resolution)
    z = np.linspace(0, domain_size[2], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 准备输入数据
    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    power_input = np.full((points.shape[0], 1), power)
    
    # 转换为PyTorch张量
    points_tensor = torch.FloatTensor(points)
    power_tensor = torch.FloatTensor(power_input)
    
    # 使用模型预测温度
    model.eval()
    with torch.no_grad():
        temperatures = model(
            points_tensor[:, 0:1],
            points_tensor[:, 1:2],
            points_tensor[:, 2:3],
            power_tensor
        )
    
    # 重塑为3D网格
    temp_grid = temperatures.numpy().reshape(resolution, resolution, resolution)
    
    # 创建3D等值面图
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=temp_grid.flatten(),
        isomin=temp_grid.min(),
        isomax=temp_grid.max(),
        colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=f'3D Temperature Field (Power: {power}W)'
    )
    
    return fig

def plot_temperature_slice(model, plane='xy', position=0.5, domain_size=(1.0, 1.0, 1.0), 
                         power=50.0, resolution=50):
    """
    绘制温度场的2D切片图
    
    参数:
    model: 训练好的PINN模型
    plane: 切片平面 ('xy', 'yz', 或 'xz')
    position: 切片位置（归一化到[0,1]）
    domain_size: 空间域大小
    power: 加热功率
    resolution: 每个维度的网格点数
    """
    # 创建2D网格
    if plane == 'xy':
        x = np.linspace(0, domain_size[0], resolution)
        y = np.linspace(0, domain_size[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, position * domain_size[2])
    elif plane == 'yz':
        y = np.linspace(0, domain_size[1], resolution)
        z = np.linspace(0, domain_size[2], resolution)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, position * domain_size[0])
    else:  # 'xz'
        x = np.linspace(0, domain_size[0], resolution)
        z = np.linspace(0, domain_size[2], resolution)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, position * domain_size[1])
    
    # 准备输入数据
    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    power_input = np.full((points.shape[0], 1), power)
    
    # 转换为PyTorch张量
    points_tensor = torch.FloatTensor(points)
    power_tensor = torch.FloatTensor(power_input)
    
    # 使用模型预测温度
    model.eval()
    with torch.no_grad():
        temperatures = model(
            points_tensor[:, 0:1],
            points_tensor[:, 1:2],
            points_tensor[:, 2:3],
            power_tensor
        )
    
    # 重塑为2D网格
    temp_grid = temperatures.numpy().reshape(resolution, resolution)
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=temp_grid,
        colorscale='Jet',
        colorbar=dict(title='Temperature')
    ))
    
    # 设置坐标轴标签
    if plane == 'xy':
        xlabel, ylabel = 'X', 'Y'
    elif plane == 'yz':
        xlabel, ylabel = 'Y', 'Z'
    else:
        xlabel, ylabel = 'X', 'Z'
    
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title=f'Temperature Field {plane.upper()}-Plane (Position: {position:.2f})'
    )
    
    return fig