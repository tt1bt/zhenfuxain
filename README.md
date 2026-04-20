# CIAH Reproduction (PatternNet)

This project follows the step-by-step pipeline in the provided notes and includes:

- ResNet34 + hash layer baseline
- Long-tail dataset construction
- Centripetal hash loss
- Optional IAM class weighting
- Retrieval evaluation with mAP / P@K / R@K / head-middle-tail splits

## Project Structure

- dataset/patternnet_dataset.py
- models/hash_model.py
- utils/longtail_dataset.py
- utils/centripetal_loss.py
- utils/iam_loss.py
- utils/retrieval.py
- train.py
- test.py

## Install

```bash
pip install -r requirements.txt
```

## Data Layout

Put dataset under:

```text
data/PatternNet/
  airplane/
  baseball_field/
  ...
```

## Train (examples)

Plain CE:

```bash
python train.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --epochs 150 --cls_weighting none --weights_out model_plain_PatternNet.pth
```

Class-balanced CE:

```bash
python train.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --epochs 150 --cls_weighting class_balanced --cb_beta 0.9999 --weights_out model_cb_PatternNet.pth
```

## Test (single)

```bash
python test.py --root data/PatternNet --imb_factor 0.01 --hash_bits 32 --weights model_plain_PatternNet.pth --topk 0 --out_tag plain
```

## Test (paper-like batch: IF x bits)

```bash
python test.py --root data/PatternNet --paper_like --weights_template "model_bits{bits}_if{imb_factor}.pth"
```

## Key Loss

Training uses:

- hash loss: centripetal loss
- cls loss: CE or IAM weighted CE
- total: `hash_loss + alpha * cls_loss`

## Notes

- `split_path` is reused across train/test for reproducible query/db split.
- If `imb_factor` or `query_ratio` changes, split file is regenerated automatically.
- `--tsne` in test exports CSV for visualization.
