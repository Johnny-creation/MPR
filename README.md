# Structured Multi-Prototype Routing for Few-Shot Medical Anomaly Detection

Official PyTorch implementation of **Structured Multi-Prototype Routing for Few-Shot Medical Anomaly Detection**.

## Environment

```bash
pip install -r requirements.txt
```

## Download

Download the packaged archive `MPR.tar.gz` and extract it in the project root to restore both `models/` and `data/` at once:

- Download link: `https://pan.baidu.com/s/1GoJt-CMRduCQ26m_cXFY0Q?pwd=1234`

```bash
tar -zxvf MPR.tar.gz
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

Or use the wrapper script:

```bash
bash scripts/train_all.sh --save_model
```

Evaluate checkpoint:

```bash
python test.py --checkpoint tmp/Brain_shot2_seed111_it1/best.pth
```

## Test Results

Evaluation results on the six medical anomaly detection datasets are reported.

`AUC_img`

| Dataset | 2-shot | 4-shot | 8-shot | 16-shot |
| --- | ---: | ---: | ---: | ---: |
| Brain | 0.9625 | 0.9646 | 0.9692 | 0.9721 |
| Chest | 0.8808 | 0.8870 | 0.8715 | 0.9011 |
| Histopathology | 0.8528 | 0.8514 | 0.8835 | 0.9123 |
| Liver | 0.8845 | 0.8903 | 0.9398 | 0.9516 |
| Retina_OCT2017 | 0.9913 | 0.9952 | 0.9991 | 0.9985 |
| Retina_RESC | 0.9731 | 0.9757 | 0.9858 | 0.9877 |
| Average | 0.9242 | 0.9274 | 0.9415 | 0.9539 |

`AUC_pixel`

| Dataset | 2-shot | 4-shot | 8-shot | 16-shot |
| --- | ---: | ---: | ---: | ---: |
| Brain | 0.9839 | 0.9831 | 0.9844 | 0.9835 |
| Liver | 0.9966 | 0.9958 | 0.9979 | 0.9985 |
| Retina_RESC | 0.9897 | 0.9909 | 0.9938 | 0.9949 |
| Average | 0.9901 | 0.9899 | 0.9920 | 0.9923 |

- Checkpoints download: `ttps://pan.baidu.com/s/1GThLCTh2y15_6GG-cSGCnA?pwd=1234`

## Outputs

Training outputs are saved to `tmp/<run_name>/`:

- `resolved_args.json`
- `metrics.csv`
- `best.json`
- `best.pth` (when `--save_model` is enabled)
