# CIAH 复现（PatternNet）

本项目按照给定笔记中的流程进行实现，主要包含：

- ResNet34 + 哈希层基线模型
- 长尾数据集构建
- 向心哈希损失
- 类平衡 IAM 类别加权
- 检索评估（mAP / P@K / R@K / 头部-中部-尾部分组）

## 项目结构

- dataset/patternnet_dataset.py
- models/hash_model.py
- utils/longtail_dataset.py
- utils/centripetal_loss.py
- utils/iam_loss.py
- utils/retrieval.py
- train.py
- test.py

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据目录

请将数据集放在如下目录结构中：

```text
data/PatternNet/
  airplane/
  baseball_field/
  ...
```

## 训练示例

普通交叉熵（CE）：

```bash
python train.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --epochs 150 --cls_weighting none --weights_out model_plain_PatternNet.pth
```

类平衡交叉熵（Class-balanced CE）：

```bash
python train.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --epochs 150 --cls_weighting class_balanced --cb_beta 0.999 --cb_mode 1-beta --weights_out model_cb_PatternNet.pth
```

类平衡交叉熵（尾部补偿更强，分子为 1）：

```bash
python train.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --epochs 150 --cls_weighting class_balanced --cb_beta 0.999 --cb_mode 1 --weights_out model_cb1_PatternNet.pth
```

## 测试（单次）

```bash
python test.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --weights model_plain_PatternNet.pth --topk 0 --out_tag plain
```

## 测试（论文风格批量：不同 IF 与 bits）

```bash
python test.py --root data/PatternNet --paper_like --weights_template "model_bits{bits}_if{imb_factor}.pth"
```

## CLRS 遥感图像检索应用

这个本地 Web 应用会自动加载 `model_cb1_CLRS_if0.05_bits32.pth`，上传一张遥感图像后，在 CLRS 数据集中检索最相似的 5 张图。

启动命令：

```bash
python app_clrs_retrieval.py
```

默认会在浏览器打开 `http://127.0.0.1:7860`。

如果想手动指定参数：

```bash
python app_clrs_retrieval.py --root data/CLRS --weights model_cb1_CLRS_if0.05_bits32.pth --split_path split_CLRS.json --device auto
```

## 核心损失

训练中使用：

- 哈希损失：中心余弦对比损失
- 分类损失：按类别权重加权的 CrossEntropy
- 总损失：`L_C1 + 0.2 * L_C2`

默认配置：

- `alpha=0.2`
- `cls_weighting=class_balanced`
- `cb_beta=0.999`
- `cb_mode=1`（当 `cls_weighting=class_balanced` 时）

## 说明

- `split_path` 会在 train/test 间复用，以保证 query/db 划分可复现。
- 当 `imb_factor` 或 `query_ratio` 变化时，会自动重新生成 split 文件。
- 在 test 中启用 `--tsne` 会导出用于可视化的 CSV 文件。
