# 项目：向心哈希与重要性感知的遥感图像长尾检索（复现）

## 一句话概述
复现实验《Hashing for Retrieving Long-Tailed Distributed Remote Sensing Images》，实现向心哈希（Centripetal Hashing）与重要性感知模块（IAM），用于长尾分布遥感图像的哈希检索与评估。

## 目标
- 复现论文中的训练流程与检索评估（mAP、P@K、PR 曲线、head/middle/tail 分组报告）。
- 提供可复现的训练/测试脚本、模型权重与简单的检索演示（Flask）。

## 技术栈
- PyTorch (+ torchvision)
- scikit-learn, matplotlib（可视化）
- Flask（演示应用）

## 数据与目录结构
项目假定的数据目录结构：

data/<数据集名>/<类别名>/<图片文件>

常用数据集：PatternNet、NWPU-RESISC45、RSSCN7、CLRS。

## 主要文件一览
- `train.py`：单数据集训练入口
- `test.py`：检索评估入口（support paper_like 批量评估）
- `models/hash_model.py`：模型定义（ResNet34 + IAM + hash layer + classifier）
- `utils/centripetal_loss.py`：向心哈希损失
- `utils/iam_loss.py`：类别平衡分类损失（CB-CE）
- `utils/retrieval.py`：检索评估与分组指标
- `app_clrs_retrieval.py`：Flask 演示应用
- `train_test_cb1_grid.py`：网格化训练/测试脚本（工作区当前文件）

## 典型运行示例
（在激活 `DL` conda 环境后运行）

```powershell
python train.py --dataset CLRS --weights out.pth --cls_weighting class_balanced --cb_beta 0.999 --alpha 0.2
python test.py --weights out.pth --topk 0
python app_clrs_retrieval.py --model out.pth
```

## 成功标准
- 能在目标数据集上复现论文表格中相近的 mAP / P@K 趋势（head/middle/tail 改善显著）。
- 提供可运行的训练/评估脚本与一个小型 Flask 演示（可加载本地模型）。

## 联系与注意事项
- 所有文档与注释使用中文。添加/修改代码请遵循仓库现有风格与最小改动原则。
