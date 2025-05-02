import numpy as np
import torch

def generate_training_data(n_points=1000, domain_size=(1.0, 1.0, 1.0), power_range=(10, 100)):
    """
    生成用于训练的3D温度场数据
    
    参数:
    n_points: 采样点数量
    domain_size: (x_size, y_size, z_size) 空间域大小
    power_range: (min_power, max_power) 加热功率范围
    """
    # 随机生成空间坐标点
    x = np.random.uniform(0, domain_size[0], (n_points, 1))
    y = np.random.uniform(0, domain_size[1], (n_points, 1))
    z = np.random.uniform(0, domain_size[2], (n_points, 1))
    
    # 生成随机功率值
    power = np.random.uniform(power_range[0], power_range[1], (n_points, 1))
    
    # 将坐标组合在一起
    coords = np.hstack([x, y, z])
    
    # 生成一些边界条件数据（这里使用简化模型）
    temperature = np.zeros((n_points, 1))
    for i in range(n_points):
        # 简化的温度计算模型，实际应用中可能需要更复杂的物理模型
        distance = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        temperature[i] = power[i] / (4 * np.pi * distance + 1e-6)
    
    # 转换为PyTorch张量
    data = {
        'coords': torch.FloatTensor(coords),
        'power': torch.FloatTensor(power),
        'temperature': torch.FloatTensor(temperature)
    }
    
    return data

def load_experimental_data(file_path):
    """
    加载实验测量的温度场数据（如果有的话）
    """
    try:
        # 这里添加实际数据加载逻辑
        data = np.load(file_path)
        return {
            'coords': torch.FloatTensor(data['coords']),
            'power': torch.FloatTensor(data['power']),
            'temperature': torch.FloatTensor(data['temperature'])
        }
    except FileNotFoundError:
        print(f"Warning: No experimental data found at {file_path}")
        return None

def create_data_loader(data, batch_size=32, shuffle=True):
    """
    创建PyTorch数据加载器
    """
    dataset = torch.utils.data.TensorDataset(
        data['coords'],
        data['power'],
        data['temperature']
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)