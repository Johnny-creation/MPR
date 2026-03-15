import argparse
import contextlib
import copy
import csv
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch

from config.train_defaults import get_advanced_defaults, get_train_defaults
from dataset.medical_few import MedDataset
from test import evaluate_model
from utils import augment
from CTR import DINO_Inplanted, load_dino_backbone
from clip.clip import create_model
from loss import ComponentLosses, BinaryDiceLoss, FocalLoss, Loss_detection
from prompt.promptChooser import PromptChooser

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_INDEX = {"Brain": 3, "Liver": 2, "Retina_RESC": 1, "Retina_OCT2017": -1, "Chest": -2, "Histopathology": -3}
ALL_DATASETS = ["Brain", "Liver", "Retina_RESC", "Retina_OCT2017", "Chest", "Histopathology"]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_args(args):
    args = copy.deepcopy(args)
    args.lam_div = float(args.lam_div)

    args.llm_prompt = bool(args.llm_prompt)
    args.component_count = max(1, int(args.component_count))

    return args


def run_dir_for(args) -> str:
    run_name = f"{args.obj}_shot{int(args.shot)}_seed{int(args.seed)}_it{int(args.iterate)}"
    out_dir = os.path.join(args.save_path, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_metrics_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_checkpoint_payload(args, model, text_chooser, loss_det, epoch, best_metrics):
    payload = {
        "epoch": int(epoch),
        "train_args": vars(args).copy(),
        "best_metrics": dict(best_metrics),
        "ctr_blocks": model.ctr_blocks.state_dict(),
        "ctr_optimizer_state": model.ctr_optimizer.state_dict(),
        "loss_det_state": loss_det.state_dict(),
    }
    payload = text_chooser.save_prompt(payload)
    return payload


def train_single_dataset(base_args, clip_model, obj_name, shot: int):
    args = copy.deepcopy(base_args)
    args.obj = obj_name
    args.shot = int(shot)
    setup_seed(args.seed)

    out_dir = run_dir_for(args)
    save_json(os.path.join(out_dir, "resolved_args.json"), vars(args))
    # print("[Resolved]")
    # print(f"obj={args.obj}")
    # print(f"shot={args.shot}")
    # print(f"component_count={args.component_count}")
    # print(f"routing=topksoft/{args.router_topk}")
    # print(f"llm_prompt={int(args.llm_prompt)}")
    # print(f"lam_diff={args.lam_diff}")
    # print(f"lam_div={args.lam_div}")
    # print(f"diff_margin={args.diff_margin}")
    # print(f"div_margin={args.div_margin}")

    if torch.cuda.is_available():
        kwargs = {"num_workers": int(args.num_workers), "pin_memory": True}
    else:
        kwargs = {"num_workers": 0}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, **kwargs)

    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat(
        [torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0
    )
    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_det = Loss_detection(args=args, device=device, loss_type=args.loss_type, dec_type=args.dec_type, lr=args.learning_rate)
    loss_component = ComponentLosses(args)
    use_component = (
        (args.lam_diff > 0)
        or (args.lam_div > 0)
    )
    has_seg = CLASS_INDEX[args.obj] > 0

    text_chooser = PromptChooser(clip_model, args, device)
    dino_backbone = load_dino_backbone(args, device)
    model = DINO_Inplanted(args, dino_backbone=dino_backbone).to(device)
    model.eval()

    best = {"best_epoch": -1, "auc_img": -1.0, "auc_pixel": -1.0, "score": -1.0, "ckpt_path": None}
    metrics_rows: List[Dict] = []

    print(f"Training on {obj_name} | out_dir: {out_dir} | batch_size: {args.batch_size}")
    for epoch in range(args.epoch):
        loss_list = []
        det_loss_list = []
        seg_loss_list = []
        component_loss_list = []
        for (image, gt, label) in train_loader:
            image = image.to(device)
            autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else contextlib.nullcontext()
            with autocast_ctx:
                text_features = text_chooser()
                _, det_model, seg_model = model(image, text_features)
                image_label = label.to(device)

                det_loss = 0
                for layer in range(len(det_model)):
                    det_loss += loss_det(det_model[layer], image_label)
                loss = det_loss

                if has_seg:
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_model)):
                        seg_scores_cur = loss_det.sync_AS(seg_model[layer])
                        seg_loss += loss_focal(seg_scores_cur, mask)
                        seg_loss += loss_dice(seg_scores_cur[:, 1, :, :].unsqueeze(1), mask)
                    loss = loss + seg_loss
                else:
                    seg_loss = torch.tensor(0.0, device=device)
                    mask = None

                if use_component:
                    component_loss_val, _ = loss_component(
                        det_logits_list=det_model,
                        seg_logits_list=seg_model if has_seg else None,
                        labels=image_label,
                        text_features=text_features,
                        mask=mask,
                    )
                    loss = loss + component_loss_val
                else:
                    component_loss_val = torch.tensor(0.0, device=device)

                model.ctr_optimizer.zero_grad()
                text_chooser.text_optimizer.zero_grad()
                loss_det.optimizer.zero_grad()
                loss.backward()
                model.ctr_optimizer.step()
                text_chooser.text_optimizer.step()
                loss_det.optimizer.step()

                loss_list.append(float(loss.item()))
                det_loss_list.append(float(det_loss.item()))
                seg_loss_list.append(float(seg_loss.item()))
                component_loss_list.append(float(component_loss_val.item()))

        should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch == args.epoch - 1)
        if not should_eval:
            print(
                f"[{obj_name}] Epoch {epoch} | loss={np.mean(loss_list):.4f} det={np.mean(det_loss_list):.4f} "
                f"seg={np.mean(seg_loss_list):.4f} component={np.mean(component_loss_list):.4f} | "
                "eval skipped"
            )
            continue

        eval_out = evaluate_model(
            args=args,
            model=model,
            test_loader=test_loader,
            text_chooser=text_chooser,
            loss_det=loss_det,
            device=device,
        )
        auc_img = float(eval_out["auc_img"])
        auc_pixel = float(eval_out["auc_pixel"]) if has_seg and eval_out["auc_pixel"] is not None else -1.0
        pauc = float(eval_out["pauc"]) if eval_out["pauc"] is not None else auc_img
        score = (auc_img + auc_pixel) * 0.5 if has_seg else auc_img

        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(loss_list)),
            "loss_det": float(np.mean(det_loss_list)),
            "loss_seg": float(np.mean(seg_loss_list)),
            "loss_component": float(np.mean(component_loss_list)),
            "auc_img": auc_img,
            "pauc": pauc,
        }
        if has_seg:
            row["auc_pixel"] = auc_pixel
        metrics_rows.append(row)
        save_metrics_csv(os.path.join(out_dir, "metrics.csv"), metrics_rows)

        is_best = score > best["score"]
        if is_best:
            best["best_epoch"] = int(epoch)
            best["auc_img"] = auc_img
            best["auc_pixel"] = auc_pixel if has_seg else -1.0
            best["score"] = score
            if bool(args.save_model):
                ckpt_path = os.path.join(out_dir, "best.pth")
                save_dict = build_checkpoint_payload(
                    args=args,
                    model=model,
                    text_chooser=text_chooser,
                    loss_det=loss_det,
                    epoch=epoch,
                    best_metrics=best,
                )
                torch.save(save_dict, ckpt_path)
                best["ckpt_path"] = ckpt_path
            save_json(os.path.join(out_dir, "best.json"), best)

        metrics_text = f"AUC_img={auc_img:.4f}"
        if has_seg:
            metrics_text += f" AUC_pixel={auc_pixel:.4f}"
        print(
            f"[{obj_name}] Epoch {epoch} | loss={row['loss']:.4f} det={row['loss_det']:.4f} "
            f"seg={row['loss_seg']:.4f} component={row['loss_component']:.4f} | "
            f"{metrics_text}"
        )
        if is_best:
            best_text = f"AUC_img: {auc_img:.4f}"
            if has_seg:
                best_text += f" | AUC_pixel: {auc_pixel:.4f}"
            print(f"BEST [{obj_name} {args.shot}-shot] @ epoch {epoch} | {best_text}")

        if int(args.patience) > 0 and best["best_epoch"] >= 0:
            if (epoch - int(best["best_epoch"])) >= int(args.patience):
                print(f"[{obj_name}] Early stopping at epoch {epoch}. patience={args.patience}")
                break

    save_json(os.path.join(out_dir, "best.json"), best)

    print(
        "Training finished. "
        f"Best epoch: {best['best_epoch']}, AUC_img: {best['auc_img']:.4f}"
        + (f", AUC_pixel: {best['auc_pixel']:.4f}" if has_seg else "")
    )

    result = {
        "dataset": obj_name,
        "shot": int(args.shot),
        "seed": int(args.seed),
        "best_epoch": int(best["best_epoch"]),
        "best_auc_img": float(best["auc_img"]),
    }
    if has_seg:
        result["best_auc_pixel"] = float(best["auc_pixel"])
    return result


def _positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
    return ivalue


def _add_args(parser: argparse.ArgumentParser, defaults: Dict):
    parser.add_argument("--model_name", type=str, default=defaults["model_name"])
    parser.add_argument("--pretrain", type=str, default=defaults["pretrain"])
    parser.add_argument("--dino_arch", type=str, default=defaults["dino_arch"])
    parser.add_argument("--dino_weights", type=str, default=defaults["dino_weights"])
    parser.add_argument("--obj", type=str, default=defaults["obj"])
    parser.add_argument("--data_path", type=str, default=defaults["data_path"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_path", type=str, default=defaults["save_path"])
    parser.add_argument("--cuda", type=str, default=defaults["cuda"])
    parser.add_argument("--img_size", type=int, default=defaults["img_size"])
    parser.add_argument("--epoch", type=int, default=defaults["epoch"])
    parser.add_argument("--eval_interval", type=_positive_int, default=defaults["eval_interval"])
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--features_list", type=int, nargs="+", default=defaults["features_list"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--shots", "--shot", type=_positive_int, nargs="+", default=defaults["shots"])
    parser.add_argument("--iterate", type=int, default=defaults["iterate"])
    parser.add_argument("--patience", type=int, default=defaults["patience"], help="Early stopping patience in epochs")
    parser.add_argument("--run_all", action="store_true")


def _add_advanced_args(parser: argparse.ArgumentParser, defaults: Dict):
    parser.add_argument("--llm_prompt", type=int, choices=[0, 1], default=defaults["llm_prompt"])
    parser.add_argument("--llm_prompt_path", type=str, default=defaults["llm_prompt_path"])
    parser.add_argument("--component_count", type=_positive_int, default=defaults["component_count"])
    parser.add_argument("--text_adapt_until", type=int, default=defaults["text_adapt_until"])
    parser.add_argument("--text_proj_trainable", type=int, choices=[0, 1], default=defaults["text_proj_trainable"])
    parser.add_argument("--n_ctx", type=_positive_int, default=defaults["n_ctx"])

    parser.add_argument("--contrast_mood", type=str, choices=["yes", "no"], default=defaults["contrast_mood"])
    parser.add_argument("--router_topk", type=_positive_int, default=defaults["router_topk"])
    parser.add_argument("--tau", type=float, default=defaults["tau"])
    parser.add_argument("--lam_diff", type=float, default=defaults["lam_diff"])
    parser.add_argument("--lam_div", type=float, default=defaults["lam_div"])
    parser.add_argument("--diff_margin", type=float, default=defaults["diff_margin"])
    parser.add_argument("--div_margin", type=float, default=defaults["div_margin"])

    parser.add_argument("--dec_type", type=str, choices=["mean", "max", "both"], default=defaults["dec_type"])
    parser.add_argument("--loss_type", type=str, choices=["softmax", "sigmoid", "both"], default=defaults["loss_type"])

    parser.add_argument("--ctr_heads", type=_positive_int, default=defaults["ctr_heads"])
    parser.add_argument("--ctr_head_dim", type=_positive_int, default=defaults["ctr_head_dim"])
    parser.add_argument("--ctr_proposal_w", type=_positive_int, default=defaults["ctr_proposal_w"])
    parser.add_argument("--ctr_proposal_h", type=_positive_int, default=defaults["ctr_proposal_h"])
    parser.add_argument("--ctr_topk", type=_positive_int, default=defaults["ctr_topk"])
    parser.add_argument("--ctr_sim_pow", type=float, default=defaults["ctr_sim_pow"])
    parser.add_argument("--ctr_layerscale_init", type=float, default=defaults["ctr_layerscale_init"])
    parser.add_argument("--ctr_norm", type=str, choices=["in", "ln", "none"], default=defaults["ctr_norm"])


def parse_args():
    parser = argparse.ArgumentParser(description="MPR Training")
    train_defaults = get_train_defaults(REPO_ROOT)
    advanced_defaults = get_advanced_defaults(REPO_ROOT)

    _add_args(parser, train_defaults)
    _add_advanced_args(parser, advanced_defaults)

    args = parser.parse_args()
    return resolve_args(args)


def _resolve_device(cuda_arg: str) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    visible_cuda_count = torch.cuda.device_count()
    requested_cuda = int(cuda_arg)
    if requested_cuda < 0 or requested_cuda >= visible_cuda_count:
        print(
            f"[Warn] Requested cuda:{requested_cuda} but only {visible_cuda_count} visible devices. "
            "Falling back to cuda:0."
        )
        requested_cuda = 0
    return torch.device(f"cuda:{requested_cuda}")


def _build_clip_model(args, runtime_device: torch.device):
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=runtime_device,
        pretrained=args.pretrain,
        require_pretrained=True,
    )
    clip_model.to(runtime_device)
    clip_model.eval()
    return clip_model


def main():
    global device
    args = parse_args()
    device = _resolve_device(args.cuda)
    print(f"Using device: {device}")

    clip_model = _build_clip_model(args, device)
    targets = ALL_DATASETS if args.run_all else [args.obj]
    shots = [int(s) for s in args.shots]

    for shot in shots:
        for obj in targets:
            train_single_dataset(args, clip_model, obj, shot=shot)


if __name__ == "__main__":
    main()
