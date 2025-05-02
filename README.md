# 3D温度场PINN模型训练项目

这个项目使用Physics-Informed Neural Networks (PINN)来学习三维温度场分布。该模型考虑了加热板的功率大小和热导率的影响。

## 项目结构

```
.
├── data/               # 数据目录
│   ├── external/      # 外部数据
│   ├── processed/     # 处理后的数据
│   └── raw/          # 原始数据
├── notebooks/         # Jupyter notebooks
├── src/              # 源代码
│   ├── data/         # 数据处理相关代码
│   ├── features/     # 特征工程相关代码
│   ├── models/       # 模型定义和训练代码
│   ├── utils/        # 工具函数
│   └── visualization/# 可视化相关代码
├── tests/            # 测试代码
├── main.py           # 主程序
└── requirements.txt  # 项目依赖
```

## 环境要求

确保您的系统已安装Python 3.7+，然后安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备训练数据：
   - 如果使用模拟数据，程序会自动生成
   - 如果有实验数据，请放在 `data/raw/` 目录下

2. 运行训练：
   ```bash
   python main.py
   ```

3. 查看结果：
   训练完成后，程序会生成以下可视化文件：
   - `temperature_field_3d.html`：3D温度场可视化
   - `temperature_field_xy.html`：XY平面温度分布
   - `temperature_field_yz.html`：YZ平面温度分布
   - `temperature_field_xz.html`：XZ平面温度分布

## 模型说明

PINN模型结合了神经网络的拟合能力和物理定律的约束。在这个项目中：

1. 物理约束：
   - 热传导方程
   - 边界条件
   - 热源功率影响

2. 网络结构：
   - 输入：空间坐标(x,y,z)和加热功率
   - 输出：温度场
   - 隐藏层：4层 [64, 64, 64, 32]
   - 激活函数：Tanh

3. 训练参数：
   - 学习率：0.001
   - 训练轮次：2000
   - 采样点数：5000

## 自定义参数

您可以通过修改 `main.py` 中的参数来调整模型：

- `domain_size`：空间域大小
- `power_range`：功率范围
- `hidden_layers`：神经网络隐藏层结构
- `n_points`：训练数据点数
- `epochs`：训练轮次
- `learning_rate`：学习率