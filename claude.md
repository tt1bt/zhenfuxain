## 项目说明

本项目为本科毕业设计，复现 IEEE TGRS 2024 论文 **"Hashing for Retrieving Long-Tailed Distributed Remote Sensing Images"**（CIAH：向心哈希 + 重要性感知模块）。

### 核心思路

在遥感图像检索中，类别分布天然呈长尾形态（少数"头部"类样本多，大量"尾部"类样本极少）。本项目通过三个核心组件解决该问题：

1. **长尾数据集构造**：按指数衰减公式 `N_i = N_max * IF^(i/(C-1))` 模拟真实长尾分布，通过不平衡因子 IF（典型值 0.1 / 0.05 / 0.01）控制尾部稀疏程度。
2. **向心哈希损失（Centripetal Loss）**：为每个类别维护可学习的哈希中心向量，通过余弦相似度 + 交叉熵推动同类样本在汉明空间中向中心聚拢。
3. **类别平衡分类损失（Class-Balanced CE）**：基于 CVPR 2019 "Effective Number of Samples" 理论，用有效样本数 `E_n = 1 - beta^n` 进行类别重加权，beta 越接近 1 则尾部补偿越强。支持三种加权模式：`none`（普通 CE）、`sqrt_inv`（平滑倒数）、`class_balanced`（有效样本数）。
4. **IAM（Importance-Aware Module）**：共享全连接 + tanh 注意力门控，增强哈希特征的判别性。

总损失：`L = L_hash + alpha * L_cls`（alpha 默认 0.2）。

### 支持的数据集

| 数据集 | 类别数 | 每类样本 | 说明 |
|--------|--------|----------|------|
| PatternNet | 38 | 800 | 遥感场景分类 |
| NWPU-RESISC45 | 45 | 700 | 大规模遥感场景 |
| RSSCN7 | 7 | 400 | 小规模场景分类 |
| CLRS | 25 | ~200-400 | 中国土地利用遥感 |

数据目录结构统一为 `data/<数据集名>/<类别名>/<图片>`。

### 项目结构

```
├── dataset/patternnet_dataset.py  # 数据集加载与 train/query/db 划分
├── models/hash_model.py           # ResNet34 + IAM + hash_layer + classifier
├── utils/
│   ├── longtail_dataset.py        # 长尾分布采样公式
│   ├── centripetal_loss.py        # 向心哈希损失（可学习类别中心）
│   ├── iam_loss.py                # 类别平衡分类损失（CB-CE）
│   └── retrieval.py               # 检索评估（mAP/P@K/R@K/PR曲线/head-middle-tail分组）
├── train.py                       # 单数据集训练入口
├── test.py                        # 检索评估入口（单次/批量 paper_like 模式）
├── train_all_datasets.py          # 多数据集批量训练调度器
├── app_clrs_retrieval.py          # Flask Web 检索演示应用
├── summarize_clrs_metrics.py      # 日志指标汇总导出 CSV
└── split_*.json                   # 数据划分文件（保证 query/db 可复现）
```

### 关键参数速查

**训练**：`--cls_weighting`（none/sqrt_inv/class_balanced）、`--cb_beta`（0.9~0.9999，越接近 1 尾部补偿越强）、`--cb_mode`（1 / 1-beta）、`--alpha`（分类损失权重）、`--gamma`（向心损失系数）。

**评估指标**：mAP、P@K（K=10/50/100）、R@K、PR 曲线、按样本频率自动分 head/middle/tail 三组报告。

**测试时参数一致性检查**：`test.py` 会从 checkpoint 读取元信息，自动校验 `imb_factor`、`query_ratio`、`hash_bits` 是否与训练一致，不一致会报错（可用 `--allow_ckpt_mismatch` 强制评估）。

### 典型实验流程

1. 单次训练：`python train.py --cls_weighting class_balanced --cb_beta 0.999 --cb_mode 1`
2. 单次评估：`python test.py --weights model.pth --topk 0`
3. 批量评估：`python test.py --paper_like`（遍历 IF=0.1/0.05/0.01 × bits=16/32/64）
4. 多数据集 + beta 遍历：PowerShell 脚本 `run_three_datasets_beta_sweep.ps1`

### 技术栈

PyTorch + torchvision（ResNet34 预训练）+ Flask（Web 演示）+ scikit-learn（t-SNE）+ matplotlib

---

## 语言要求

- 所有对话使用中文。
- 所有文档使用中文。
- 所有代码注释使用中文。

## 执行要求

- 在生成说明、总结、计划、提交说明时，统一使用中文。
- 在新增或修改 Markdown 文档时，统一使用中文。
- 在新增或修改代码注释时，统一使用中文。
- ## 四个原则详解

### 1. 编码前思考

**不要假设。不要隐藏困惑。呈现权衡。**

LLM 经常默默选择一种解释然后执行。这个原则强制明确推理：

- **明确说明假设** — 如果不确定，询问而不是猜测
- **呈现多种解释** — 当存在歧义时，不要默默选择
- **适时提出异议** — 如果存在更简单的方法，说出来
- **困惑时停下来** — 指出不清楚的地方并要求澄清

### 2. 简洁优先

**用最少的代码解决问题。不要过度推测。**

对抗过度工程的倾向：

- 不要添加要求之外的功能
- 不要为一次性代码创建抽象
- 不要添加未要求的"灵活性"或"可配置性"
- 不要为不可能发生的场景做错误处理
- 如果 200 行代码可以写成 50 行，重写它

**检验标准：** 资深工程师会觉得这过于复杂吗？如果是，简化。

### 3. 精准修改

**只碰必须碰的。只清理自己造成的混乱。**

编辑现有代码时：

- 不要"改进"相邻的代码、注释或格式
- 不要重构没坏的东西
- 匹配现有风格，即使你更倾向于不同的写法
- 如果注意到无关的死代码，提一下 —— 不要删除它

当你的改动产生孤儿代码时：

- 删除因你的改动而变得无用的导入/变量/函数
- 不要删除预先存在的死代码，除非被要求

**检验标准：** 每一行修改都应该能直接追溯到用户的请求。

### 4. 目标驱动执行

**定义成功标准。循环验证直到达成。**