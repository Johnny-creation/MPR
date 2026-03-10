from __future__ import annotations

import os
from typing import Any, Dict


def get_train_defaults(repo_root: str) -> Dict[str, Any]:
    return {
        "model_name": "ViT-L-14-336",
        "pretrain": "openai",
        "dino_arch": "vitl16",
        "dino_weights": os.path.join(repo_root, "models", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        "obj": "Brain",
        "data_path": os.path.join(repo_root, "data"),
        "batch_size": 16,
        "num_workers": 4,
        "save_path": "./tmp",
        "cuda": "0",
        "img_size": 240,
        "epoch": 100,
        "eval_interval": 1,
        "learning_rate": 0.001,
        "features_list": [6, 12, 18, 24],
        "seed": 111,
        "shots": [2],
        "iterate": 0,
        "patience": 0,
    }


def get_advanced_defaults(repo_root: str) -> Dict[str, Any]:
    return {
        "text_adapt_until": 0,
        "text_proj_trainable": 0,
        "llm_prompt_path": os.path.join(repo_root, "prompt", "llm_prompt.json"),
        "llm_prompt": 1,
        "component_count": 6,
        "contrast_mood": "yes",
        "router_topk": 3,
        "n_ctx": 8,
        "tau": 0.7,
        "lam_diff": 2.0,
        "lam_div": 0.5,
        "div_margin": 0.85,
        "diff_margin": 0.1,
        "dec_type": "mean",
        "loss_type": "softmax",
        "ctr_heads": 4,
        "ctr_head_dim": 24,
        "ctr_proposal_w": 6,
        "ctr_proposal_h": 6,
        "ctr_topk": 1,
        "ctr_sim_pow": 1.0,
        "ctr_layerscale_init": 0.02,
        "ctr_norm": "in",
    }
