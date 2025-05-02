import torch
import numpy as np
from src.models.train_model import PINN3DTemperature, train_pinn
from src.data.load_data import generate_training_data, create_data_loader
from src.visualization.visualize import plot_3d_temperature_field, plot_temperature_slice

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义问题域参数
    domain_size = (1.0, 1.0, 1.0)  # 1m x 1m x 1m 空间
    power_range = (10, 100)  # 10-100W
    
    # 生成训练数据
    print("生成训练数据...")
    train_data = generate_training_data(
        n_points=5000,
        domain_size=domain_size,
        power_range=power_range
    )
    
    # 创建PINN模型
    print("初始化PINN模型...")
    model = PINN3DTemperature(hidden_layers=[64, 64, 64, 32])
    
    # 训练模型
    print("开始训练模型...")
    train_pinn(
        model=model,
        x_train=train_data,
        y_train={'temperature': train_data['temperature']},
        epochs=2000,
        learning_rate=1e-3
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'model_weights.pth')
    print("模型已保存到 'model_weights.pth'")
    
    # 生成可视化
    print("生成温度场可视化...")
    
    # 3D温度场可视化
    fig_3d = plot_3d_temperature_field(
        model,
        domain_size=domain_size,
        power=50.0,
        resolution=20
    )
    fig_3d.write_html('temperature_field_3d.html')
    
    # 在不同平面上生成2D切片图
    planes = ['xy', 'yz', 'xz']
    for plane in planes:
        fig_2d = plot_temperature_slice(
            model,
            plane=plane,
            position=0.5,
            domain_size=domain_size,
            power=50.0,
            resolution=50
        )
        fig_2d.write_html(f'temperature_field_{plane}.html')
    
    print("可视化结果已保存为HTML文件")

if __name__ == '__main__':
    main()