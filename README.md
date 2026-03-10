# Structured Multi-Prototype Routing for Few-Shot Medical Anomaly Detection

Official PyTorch implementation of **Structured Multi-Prototype Routing for Few-Shot Medical Anomaly Detection**.

## Environment

```bash
pip install -r requirements.txt
```

## Pretrained Models

Place the following files under `models/`:

- `ViT-L-14-336px.pt`
- `ViT-L-14-336.json`
- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

## Dataset

Use dataset root: `data/{Dataset}_AD/`.

Example layout:

```text
data/Brain_AD/
  valid/good/img
  valid/Ungood/img
  valid/Ungood/anomaly_mask
  test/good/img
  test/Ungood/img
  test/Ungood/anomaly_mask
```

Few-shot split files:

```text
dataset/fewshot_seed/{Dataset}/{shot}-shot.txt
```

## Quick Start

Train on one dataset:

```bash
python train.py --save_model --obj Brain --shot 2
```

Train on all datasets and all shots:

```bash
python train.py --save_model --run_all --shot 2 4 8 16
```

Evaluate checkpoint:

```bash
python test.py --checkpoint tmp/Brain_shot2_seed111_it1/best.pth
```

## Outputs

Training outputs are saved to `tmp/<run_name>/`:

- `resolved_args.json`
- `metrics.csv`
- `best.json`
- `best.pth` (when `--save_model` is enabled)
