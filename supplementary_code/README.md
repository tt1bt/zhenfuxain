# 支持材料代码

复现论文：IEEE TGRS 2024 "Hashing for Retrieving Long-Tailed Distributed Remote Sensing Images"（CIAH：向心哈希 + 重要性感知模块）

## 环境依赖

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch 1.12+
- torchvision

## 目录结构

```
├── models/
│   └── hash_model.py          # ResNet34 主干 + IAM + hash_layer + 分类器
├── utils/
│   ├── centripetal_loss.py    # 向心哈希损失（可学习类别中心）
│   ├── iam_loss.py            # 类别平衡交叉熵损失（有效样本数加权）
│   ├── longtail_dataset.py    # 长尾分布采样公式
│   └── retrieval.py           # 检索评估指标（mAP、P@K、R@K、PR 曲线）
├── dataset/
│   └── patternnet_dataset.py  # 数据加载与 train/query/db 划分
├── train.py                   # 训练入口
├── test.py                    # 检索评估入口
├── train_all_datasets.py      # 多数据集批量调度器
└── requirements.txt           # 依赖清单
```

## 使用方法

### 训练

```bash
# 单次训练
python train.py --cls_weighting class_balanced --cb_beta 0.999 --cb_mode 1

# 主要参数
#   --cls_weighting: none | sqrt_inv | class_balanced
#   --cb_beta:       0.9~0.9999（越接近 1 尾部补偿越强）
#   --cb_mode:       1 | 1-beta
#   --alpha:         分类损失权重（默认 0.2）
#   --gamma:         向心损失系数
```

### 评估

```bash
# 单次评估
python test.py --weights model.pth --topk 0

# 批量评估（IF=0.1/0.05/0.01 × bits=16/32/64）
python test.py --paper_like
```

### 支持的数据集

| 数据集 | 类别数 | 每类样本数 |
|--------|--------|------------|
| PatternNet | 38 | 800 |
| NWPU-RESISC45 | 45 | 700 |
| RSSCN7 | 7 | 400 |
| CLRS | 25 | ~200-400 |

数据目录结构：`data/<数据集名>/<类别名>/<图片>`

## 核心组件

1. **向心哈希损失（Centripetal Loss）**：为每个类别维护可学习的哈希中心向量，通过余弦相似度 + 交叉熵推动同类样本在汉明空间中向中心聚拢。

2. **类别平衡分类损失（Class-Balanced CE）**：基于有效样本数 $E_n = 1 - \beta^n$ 进行类别重加权，$\beta$ 越接近 1 尾部补偿越强。

3. **重要性感知模块（IAM）**：共享全连接 + tanh 注意力门控，增强哈希特征的判别性。

总损失：$L = L_{hash} + \alpha \cdot L_{cls}$
