import argparse
import contextlib
import json
import os
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset.medical_few import MedDataset
from CTR import DINO_Inplanted, load_dino_backbone
from clip.clip import create_model
from loss import Loss_detection
from prompt.promptChooser import PromptChooser

CLASS_INDEX = {"Brain": 3, "Liver": 2, "Retina_RESC": 1, "Retina_OCT2017": -1, "Chest": -2, "Histopathology": -3}


def _safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray, name: str) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError as e:
        print(f"[AUC] {name}: roc_auc_score fallback to 0.5 due to: {e}")
        return 0.5


def evaluate_model(
    args,
    model,
    test_loader,
    text_chooser,
    loss_det,
    device,
) -> Dict[str, Optional[float]]:
    gt_list = []
    gt_mask_list = []
    det_final = []
    seg_final = []

    with torch.no_grad():
        text_features = text_chooser()

    for (image, y, mask) in test_loader:
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else contextlib.nullcontext()
        with torch.no_grad(), autocast_ctx:
            _, det_model, seg_model = model(image, text_features)

            if CLASS_INDEX[args.obj] > 0:
                anomaly_maps = []
                for layer in range(len(seg_model)):
                    seg_scores_cur = loss_det.sync_AS(seg_model[layer])
                    anomaly_map = 0.5 * (1 - seg_scores_cur[:, 0, :, :]) + 0.5 * seg_scores_cur[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map = np.sum(np.stack(anomaly_maps), axis=0)
                seg_final.extend(score_map)

            anomaly_scores_all = 0
            for layer in range(len(det_model)):
                anomaly_scores_all += loss_det.validation(det_model[layer])
            det_final.extend(anomaly_scores_all.cpu().numpy())
            gt_mask_list.extend(mask.squeeze(1).cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = (np.asarray(gt_mask_list) > 0).astype(np.int_)

    out = {
        "auc_img": 0.0,
        "auc_pixel": None,
        "pauc": None,
        "combined": 0.0,
    }

    if CLASS_INDEX[args.obj] > 0:
        seg_scores = np.array(seg_final)
        seg_scores = (seg_scores - seg_scores.min()) / (1e-4 + seg_scores.max() - seg_scores.min())
        auc_pixel = _safe_roc_auc_score(
            gt_mask_list.flatten(),
            seg_scores.flatten(),
            name="pixel",
        )
        image_scores = np.max(seg_scores.reshape(seg_scores.shape[0], -1), axis=1)
        auc_img = _safe_roc_auc_score(
            gt_list,
            image_scores,
            name="image",
        )
        out["auc_img"] = float(auc_img)
        out["auc_pixel"] = float(auc_pixel)
        out["pauc"] = float(auc_pixel)
        out["combined"] = float(auc_img + auc_pixel)
    else:
        det_scores = np.array(det_final)
        det_scores = (det_scores - det_scores.min()) / (1e-4 + det_scores.max() - det_scores.min())
        auc_img = _safe_roc_auc_score(
            gt_list,
            det_scores,
            name="image",
        )
        out["auc_img"] = float(auc_img)
        out["combined"] = float(auc_img)
    return out


def _merge_args(ckpt_args, cli_args):
    merged = dict(ckpt_args)
    if cli_args.obj is not None:
        merged["obj"] = cli_args.obj
    if cli_args.shot is not None:
        merged["shot"] = int(cli_args.shot)
    if cli_args.iterate is not None:
        merged["iterate"] = int(cli_args.iterate)
    if cli_args.data_path is not None:
        merged["data_path"] = cli_args.data_path
    if cli_args.batch_size is not None:
        merged["batch_size"] = int(cli_args.batch_size)
    if cli_args.num_workers is not None:
        merged["num_workers"] = int(cli_args.num_workers)
    merged["cuda"] = str(cli_args.cuda)
    merged.setdefault("component_count", int(merged.get("component_count", 6) or 6))
    return SimpleNamespace(**merged)


def _build_model_and_text(args, device):
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True,
    )
    clip_model.to(device)
    clip_model.eval()

    text_chooser = PromptChooser(clip_model, args, device)
    dino_backbone = load_dino_backbone(args, device)
    model = DINO_Inplanted(args, dino_backbone=dino_backbone).to(device)
    model.eval()
    return model, text_chooser


def _load_checkpoint_state(ckpt, model, text_chooser, loss_det):
    if "ctr_blocks" in ckpt:
        model.ctr_blocks.load_state_dict(ckpt["ctr_blocks"], strict=True)
    if "text_component_bank" in ckpt and hasattr(text_chooser, "component_bank"):
        text_chooser.component_bank.load_state_dict(ckpt["text_component_bank"], strict=True)
    if "loss_det_state" in ckpt:
        loss_det.load_state_dict(ckpt["loss_det_state"], strict=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MPR checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file path")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device id")
    parser.add_argument("--obj", type=str, default=None, help="Dataset name")
    parser.add_argument("--shot", type=int, default=None, help="Shot number")
    parser.add_argument("--iterate", type=int, default=None, help="Few-shot split index")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset root path")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Data loader workers")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "train_args" not in ckpt:
        raise KeyError("Checkpoint missing train_args. Please use checkpoints generated by train.py.")

    run_args = _merge_args(ckpt["train_args"], args)

    device = torch.device(f"cuda:{run_args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating dataset={run_args.obj}, shot={run_args.shot}, iterate={run_args.iterate}")

    model, text_chooser = _build_model_and_text(run_args, device)
    loss_det = Loss_detection(
        args=run_args,
        device=device,
        loss_type=run_args.loss_type,
        dec_type=run_args.dec_type,
        lr=run_args.learning_rate,
    )
    _load_checkpoint_state(ckpt, model, text_chooser, loss_det)

    kwargs = {"num_workers": int(run_args.num_workers), "pin_memory": True} if torch.cuda.is_available() else {"num_workers": 0}
    test_dataset = MedDataset(run_args.data_path, run_args.obj, run_args.img_size, int(run_args.shot), int(run_args.iterate))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(run_args.batch_size) * 2, shuffle=False, **kwargs)

    result = evaluate_model(run_args, model, test_loader, text_chooser, loss_det, device=device)
    metrics_text = f"Checkpoint eval | AUC_img: {result['auc_img']:.4f}"
    if CLASS_INDEX[run_args.obj] > 0 and result["auc_pixel"] is not None:
        metrics_text += f" | AUC_pixel: {result['auc_pixel']:.4f}"
    print(metrics_text)

    out_json = os.path.join(os.path.dirname(args.checkpoint), "test_eval.json")
    eval_summary = {
        "auc_img": result["auc_img"],
    }
    if CLASS_INDEX[run_args.obj] > 0 and result["auc_pixel"] is not None:
        eval_summary["auc_pixel"] = result["auc_pixel"]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation json: {out_json}")


if __name__ == "__main__":
    main()
