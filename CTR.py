import math
import os
import sys
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from routing import anomaly_router


def _pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # x1: [B, M, D], x2: [B, N, D] -> [B, M, N]
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


class _CTRTokenMixer(nn.Module):
    """
    CTR token mixer for square token maps.
    Input/Output: [tokens, batch, channels].
    """

    def __init__(self, c_in, heads=4, head_dim=24, proposal_w=4, proposal_h=4, topk=2, sim_pow=2.0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.topk = max(1, int(topk))
        self.sim_pow = float(sim_pow)
        self.f = nn.Conv2d(c_in, heads * head_dim, kernel_size=1)
        self.v = nn.Conv2d(c_in, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, c_in, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))

    def forward(self, x):
        # x: [T, B, C]
        t, b, c = x.shape
        side = int(math.sqrt(t))
        if side * side != t:
            raise ValueError(f"CTR expects square token count, got tokens={t}")

        x2d = x.permute(1, 2, 0).contiguous().view(b, c, side, side)  # [B, C, H, W]
        q = self.f(x2d)
        value = self.v(x2d)

        q = q.view(b, self.heads, self.head_dim, side, side).reshape(b * self.heads, self.head_dim, side, side)
        value = value.view(b, self.heads, self.head_dim, side, side).reshape(b * self.heads, self.head_dim, side, side)

        centers = self.centers_proposal(q)  # [BH, D, pw, ph]
        value_centers = self.centers_proposal(value).flatten(2).transpose(1, 2)  # [BH, M, D]

        centers_tokens = centers.flatten(2).transpose(1, 2)  # [BH, M, D]
        q_tokens = q.flatten(2).transpose(1, 2)  # [BH, N, D]
        value_tokens = value.flatten(2).transpose(1, 2)  # [BH, N, D]

        sim = torch.sigmoid(self.sim_beta + self.sim_alpha * _pairwise_cos_sim(centers_tokens, q_tokens))  # [BH, M, N]
        k = min(self.topk, sim.size(1))
        sim_max_idx = sim.topk(k=k, dim=1).indices.contiguous()
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.0)
        sim = sim * mask
        if self.sim_pow != 1.0:
            sim = sim.pow(self.sim_pow)
        sim = sim / (sim.sum(dim=1, keepdim=True) + 1e-6)

        centers_out = ((value_tokens.unsqueeze(1) * sim.unsqueeze(-1)).sum(dim=2) + value_centers) / (
            sim.sum(dim=-1, keepdim=True) + 1.0
        )  # [BH, M, D]
        out_tokens = (centers_out.unsqueeze(2) * sim.unsqueeze(-1)).sum(dim=1)  # [BH, N, D]

        out = out_tokens.transpose(1, 2).contiguous().view(b * self.heads, self.head_dim, side, side)
        out = out.view(b, self.heads * self.head_dim, side, side)
        out = self.proj(out)  # [B, C, H, W]
        out = out.view(b, c, t).permute(2, 0, 1).contiguous()  # [T, B, C]
        return out


class CTR(nn.Module):
    def __init__(
        self,
        c_in,
        bottleneck=768,
        heads=4,
        head_dim=24,
        proposal_w=4,
        proposal_h=4,
        topk=2,
        sim_pow=2.0,
        layerscale_init=1e-2,
        norm_type="in",
    ):
        super().__init__()
        self.token_mixer = _CTRTokenMixer(
            c_in=c_in,
            heads=heads,
            head_dim=head_dim,
            proposal_w=proposal_w,
            proposal_h=proposal_h,
            topk=topk,
            sim_pow=sim_pow,
        )
        self.gamma = nn.Parameter(torch.tensor(float(layerscale_init)))
        norm_type = str(norm_type).lower()
        self.norm_type = norm_type
        if norm_type == "in":
            self.norm1 = nn.InstanceNorm1d(c_in)
        elif norm_type == "ln":
            self.norm1 = nn.LayerNorm(c_in)
        elif norm_type == "none":
            self.norm1 = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type for CTR: {norm_type}")
        self.fc2 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        # x: [tokens, batch, c_in]
        x = x + self.gamma * self.token_mixer(x)
        if self.norm_type == "in":
            x_norm = x.permute(1, 2, 0)
            x_norm = self.norm1(x_norm)
            x = x_norm.permute(2, 0, 1)
        else:
            x = self.norm1(x)
        y = self.fc2(x)
        z = self.fc3(x)
        return y, z


def load_dino_backbone(args, device):
    repo_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dinov3")
    if not (os.path.isdir(repo_root) and os.path.isdir(os.path.join(repo_root, "hub"))):
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from dinov3.hub import backbones

    arch = args.dino_arch.lower()
    weights_path = os.path.abspath(args.dino_weights)

    builder = {
        "vitl16": backbones.dinov3_vitl16,
        "vitl16plus": backbones.dinov3_vitl16plus,
        "vitb16": backbones.dinov3_vitb16,
        "vits16": backbones.dinov3_vits16,
    }.get(arch)

    if builder is None:
        raise ValueError(f"Unsupported DINO architecture {args.dino_arch}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    model = builder(pretrained=True, weights=weights_path, check_hash=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class DINO_Inplanted(nn.Module):
    def __init__(self, args, dino_model=None, dino_backbone=None):
        super().__init__()
        if dino_model is None:
            dino_model = dino_backbone
        if dino_model is None:
            raise ValueError("A DINO backbone must be provided.")
        self.backbone = dino_model
        self.features: List[int] = sorted(args.features_list)
        self.embed_dim = getattr(dino_model, "embed_dim", 1024)

        heads = int(getattr(args, "ctr_heads", 4))
        head_dim = int(getattr(args, "ctr_head_dim", 24))
        proposal_w = int(getattr(args, "ctr_proposal_w", 6))
        proposal_h = int(getattr(args, "ctr_proposal_h", 6))
        topk = int(getattr(args, "ctr_topk", 1))
        sim_pow = float(getattr(args, "ctr_sim_pow", 1.0))
        layerscale_init = float(getattr(args, "ctr_layerscale_init", 0.02))
        norm_type = str(getattr(args, "ctr_norm", "in"))

        ctr_block_ctor = lambda: CTR(
            self.embed_dim,
            bottleneck=768,
            heads=heads,
            head_dim=head_dim,
            proposal_w=proposal_w,
            proposal_h=proposal_h,
            topk=topk,
            sim_pow=sim_pow,
            layerscale_init=layerscale_init,
            norm_type=norm_type,
        )
        self.ctr_blocks = nn.ModuleList([ctr_block_ctor() for _ in self.features])
        self.ctr_optimizer = torch.optim.Adam(
            [{"params": self.ctr_blocks.parameters(), "lr": args.learning_rate}],
            betas=(0.5, 0.999),
        )

        self.contrast_mood = args.contrast_mood
        if self.contrast_mood == "no":
            self.contrast = lambda a, b: a
        elif self.contrast_mood == "yes":
            self.contrast = lambda a, b: (a - b)
        else:
            raise ValueError(f"Invalid contrast_mood: {args.contrast_mood}")

        self.router_topk = int(getattr(args, "router_topk", 3))
        self.tau = float(getattr(args, "tau", 1.0))
        self.component_count = int(getattr(args, "component_count", 6))
        self.feature_indices = [f - 1 for f in self.features]

    def forward(self, x, text_features):
        with torch.no_grad():
            intermediate = self.backbone.get_intermediate_layers(
                x,
                n=self.feature_indices,
                reshape=False,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )

        det_scores = []
        seg_scores = []
        for idx, tokens in enumerate(intermediate):
            tokens = tokens.permute(1, 0, 2)
            normal_f_det_i, normal_f_seg_i = self.ctr_blocks[idx](tokens)

            normal_f_det_i = normal_f_det_i.permute(1, 0, 2)
            normal_f_seg_i = normal_f_seg_i.permute(1, 0, 2)

            normal_f_det_i = normal_f_det_i / normal_f_det_i.norm(dim=-1, keepdim=True)
            normal_f_seg_i = normal_f_seg_i / normal_f_seg_i.norm(dim=-1, keepdim=True)

            t_n, t_a = self._split_text_features(text_features)
            sim_det_normal, sim_det_abnormal, _, _ = self._pair_logits_from_single_branch(normal_f_det_i, t_n, t_a)
            det_scores.append(torch.cat([sim_det_normal, sim_det_abnormal], dim=-1))

            sim_seg_normal, sim_seg_abnormal, _, _ = self._pair_logits_from_single_branch(normal_f_seg_i, t_n, t_a)
            seg_scores.append(torch.cat([sim_seg_normal, sim_seg_abnormal], dim=-1))

        return None, det_scores, seg_scores

    def _split_text_features(self, text_features: torch.Tensor):
        if text_features.ndim != 2:
            raise ValueError(f"text_features must be [C,2K], got {tuple(text_features.shape)}")
        expected = 2 * self.component_count
        if text_features.size(1) != expected:
            raise ValueError(
                f"text_features second dim must be exactly 2K={expected}, got {tuple(text_features.shape)}"
            )
        t_n = text_features[:, : self.component_count]
        t_a = text_features[:, self.component_count : expected]
        return t_n, t_a

    def _pair_logits_from_single_branch(self, features: torch.Tensor, t_n: torch.Tensor, t_a: torch.Tensor):
        sim_n_k = (features @ t_n) / self.tau
        sim_n, w_n = anomaly_router(sim_n_k, router_topk=self.router_topk)
        sim_a_k = (features @ t_a) / self.tau
        sim_a, w_a = anomaly_router(sim_a_k, router_topk=self.router_topk)

        if self.contrast_mood == "no":
            return sim_n, sim_a, w_n, w_a
        return sim_n - sim_a, sim_a - sim_n, w_n, w_a
