from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.nn import functional as F


def anomaly_router(
    sim_a_k: torch.Tensor,
    router_topk: int = 3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        sim_a_k: [..., K] similarity scores to K anomaly components
        router_topk:
            softmax over top-k components (or fewer if K<topk)

    Returns:
        sim_a: [..., 1] routed anomaly similarity
        w: [..., K] routing weights
    """
    if sim_a_k.ndim < 1:
        raise ValueError(f"sim_a_k must have at least 1 dim, got shape={tuple(sim_a_k.shape)}")
    if sim_a_k.size(-1) < 1:
        raise ValueError(f"Last dim K must be >= 1, got shape={tuple(sim_a_k.shape)}")

    if int(router_topk) < 1:
        raise ValueError(f"router_topk must be >=1, got {router_topk}")
    k = min(int(router_topk), sim_a_k.size(-1))
    top_vals, top_idx = torch.topk(sim_a_k, k=k, dim=-1)
    masked = torch.full_like(sim_a_k, -float("inf"))
    masked.scatter_(-1, top_idx, top_vals)
    w = F.softmax(masked, dim=-1)
    sim_a = (w * sim_a_k).sum(dim=-1, keepdim=True)
    return sim_a, w
