# 边缘端交通语义场景图构建系统

> **基于数据驱动的隐式关系发现** — 利用 GNN + Transformer + 演化门控残差，在连续特征空间中自主发现交通实体的交互关系。

## 项目架构

```
Project1/
├── Dockerfile                          # Docker 镜像定义
├── docker-compose.yml                  # 多服务编排 (train/inference/test)
├── requirements.txt                    # Python 依赖
├── pyproject.toml                      # 项目元数据
├── configs/
│   └── default.yaml                    # 超参数配置
├── data/
│   └── sample/
│       └── detections.csv              # 模拟检测框数据 (需生成)
├── src/traffic_scene_graph/
│   ├── data/
│   │   └── dataset.py                  # 数据加载 & 节点特征工程
│   ├── utils/
│   │   ├── kalman_extrapolator.py      # 卡尔曼滤波外推
│   │   └── graph_builder.py            # 稀疏图构建 (IoU + 距离)
│   ├── models/
│   │   ├── edge_transformer.py         # Edge Transformer Encoder
│   │   ├── gated_residual.py           # 演化门控残差
│   │   └── scene_graph_model.py        # 完整场景图模型
│   └── training/
│       ├── contrastive_loss.py         # 时空对比学习损失
│       └── trainer.py                  # 训练循环 & TensorBoard
├── scripts/
│   ├── generate_sample_data.py         # 生成模拟数据
│   ├── train.py                        # 训练入口
│   └── inference.py                    # 推理入口
└── tests/
    ├── test_dataset.py                 # 数据模块测试
    ├── test_graph_builder.py           # 建图模块测试
    ├── test_model.py                   # 模型测试
    └── test_e2e.py                     # 端到端测试
```

## 快速开始

### 1. 生成模拟数据

```bash
python scripts/generate_sample_data.py
```

### 2. 本地运行训练

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train.py --config configs/default.yaml
```

### 3. Docker 方式

```bash
# 构建镜像
docker compose build

# 运行训练
docker compose run train

# 运行推理
docker compose run inference

# 运行测试
docker compose run test
```

## 核心技术

| 层级 | 模块 | 技术 |
|------|------|------|
| 感知层 | 节点特征 | `[类别One-Hot, x, y, w, h, vx, vy]` 11维 |
| 建图层 | 稀疏连边 | 卡尔曼外推 + 未来 bbox IoU + 距离阈值 |
| 关系学习层 | Edge Transformer | 多头自注意力 → 连续关系向量 |
| 时序融合 | 门控残差 | `E^(t) = E^(t-1) + σ(W·ΔF) ⊙ Transformer(·)` |
| 训练策略 | 对比学习 | InfoNCE + 时间正样本 + 扰动负样本 |

## 配置说明

主要超参数在 `configs/default.yaml` 中，包括：
- **数据**: CSV 路径、帧尺寸、类别数
- **建图**: 卡尔曼外推步数、IoU/距离阈值
- **模型**: Transformer 层数/头数、关系向量维度、门控隐层维度
- **训练**: 学习率、batch size、对比学习温度系数、正样本窗口
