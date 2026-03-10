from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from prompt.CoOp import PromptLearner, TextEncoder


@dataclass(frozen=True)
class TextComponentBankOutput:
    t_n: torch.Tensor  # [C]
    t_a: torch.Tensor  # [K, C]
    t_n_components: Optional[torch.Tensor] = None  # [K, C]


class TextComponentBank(nn.Module):
    """Produces K normal text components and K anomaly text components."""

    def __init__(
        self,
        clip_model,
        obj_name: str,
        device: torch.device,
        *,
        dataset_key: Optional[str] = None,
        component_count: int = 6,
        n_ctx: int = 8,
        class_token_position: Optional[List[str]] = None,
        text_adapt_until: int = 0,
        text_adapt_weight: float = 0.01,
        text_proj_trainable: bool = True,
        llm_prompt: bool = False,
        llm_prompt_path: str = "",
    ) -> None:
        super().__init__()
        if component_count < 1:
            raise ValueError(f"component_count must be >= 1, got {component_count}")
        self.component_count = int(component_count)
        self.obj_name = obj_name
        self.dataset_key = dataset_key
        self.abnormal_component_count = self.component_count
        self.llm_prompt = bool(llm_prompt)
        self.llm_prompt_path = str(llm_prompt_path) if llm_prompt_path else ""

        if class_token_position is None:
            class_token_position = ["end", "front", "middle"]

        prompt_normal = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage",
        ]
        prompt_normal = prompt_normal[: self.component_count]
        prompt_abnormal_all = [
            "damaged {}",
            "broken {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
            " disease {}",
            "abnormal {}",
        ]
        prompt_abnormal = prompt_abnormal_all[: self.abnormal_component_count]

        prompted_state_normal = [state.format(obj_name) for state in prompt_normal]
        default_prompted_state_normal = list(prompted_state_normal)
        prompted_state_abnormal = [state.format(obj_name) for state in prompt_abnormal]
        self._prompted_state_normal = list(prompted_state_normal)
        self._llm_groups_normal = None
        self._llm_groups_abnormal = None
        self._llm_groups_background = None
        self._llm_groups_normal_list: List[List[str]] = []
        self._llm_groups_abnormal_list: List[List[str]] = []
        self._llm_groups_background_list: List[List[str]] = []
        self._llm_sentences_normal: List[str] = []
        self._llm_sentences_abnormal: List[str] = []
        self._llm_sentences_background: List[str] = []
        self._normal_group_ids: List[int] = []
        self._abnormal_group_ids: List[int] = []
        if self.llm_prompt:
            self._load_llm_groups()

        if self.llm_prompt and self._llm_sentences_abnormal:
            prompted_state_abnormal, self._abnormal_group_ids = self._build_grouped_prompt_bank(
                grouped_sentences=self._llm_groups_abnormal_list,
                fallback_sentences=self._llm_sentences_abnormal,
                k=self.abnormal_component_count,
            )

        if self.llm_prompt and (self._llm_sentences_normal or self._llm_sentences_background):
            normal_pool = list(dict.fromkeys(self._llm_sentences_normal + self._llm_sentences_background))
            normal_groups = self._llm_groups_normal_list + self._llm_groups_background_list
            k_normal = self.component_count
            prompted_state_normal, self._normal_group_ids = self._build_grouped_prompt_bank(
                grouped_sentences=normal_groups,
                fallback_sentences=normal_pool,
                k=k_normal,
            )
            if len(prompted_state_normal) == 0:
                prompted_state_normal = default_prompted_state_normal
                self._normal_group_ids = []
            self._prompted_state_normal = list(prompted_state_normal)

        self.text_encoder = TextEncoder(
            clip_model,
            text_adapt_until=text_adapt_until,
            text_adapt_weight=text_adapt_weight,
            text_proj_trainable=text_proj_trainable,
        ).to(device)
        text_width = self._text_width(clip_model)
        self.g_t = nn.Linear(text_width, text_width, bias=False).to(device)
        with torch.no_grad():
            self.g_t.weight.copy_(torch.eye(text_width, device=device))

        self.abnormal_prompt_learner = PromptLearner(
            prompts={"abnormal": prompted_state_abnormal},
            n_ctx=n_ctx,
            CSC=True,
            class_token_position=class_token_position,
            clip_model=clip_model,
        ).to(device)

        self.normal_prompt_learner = PromptLearner(
            prompts={"normal": prompted_state_normal},
            n_ctx=n_ctx,
            CSC=True,
            class_token_position=class_token_position,
            clip_model=clip_model,
        ).to(device)

        self.class_token_position = list(class_token_position)

    def forward(self) -> TextComponentBankOutput:
        def _encode_prompt_learner(pl, key: str):
            prompts = pl()
            tokenized = pl.tokenized_prompts[key].repeat(len(self.class_token_position), 1)
            feats = self.text_encoder(prompts[key], tokenized)  # [P*num_prompts, C]
            P = len(self.class_token_position)
            num_prompts = pl.tokenized_prompts[key].shape[0]
            return feats.view(P, num_prompts, -1).mean(dim=0)  # [num_prompts, C]

        def _select_k(feats: torch.Tensor, k: int):
            if k <= feats.shape[0]:
                return feats[:k]
            reps = (k + feats.shape[0] - 1) // feats.shape[0]
            return feats.repeat(reps, 1)[:k]

        def _group_mean_by_ids(feats: torch.Tensor, group_ids: List[int], k: int) -> torch.Tensor:
            if len(group_ids) != feats.shape[0]:
                raise ValueError(
                    f"group_ids length ({len(group_ids)}) must match feats count ({feats.shape[0]})"
                )
            out = []
            for gid in range(k):
                idx = [i for i, g in enumerate(group_ids) if g == gid]
                if len(idx) == 0:
                    continue
                out.append(feats[idx].mean(dim=0))
            if len(out) == 0:
                return _select_k(feats, k)
            if len(out) < k:
                reps = (k + len(out) - 1) // len(out)
                out = (out * reps)[:k]
            return torch.stack(out[:k], dim=0)

        feats_n = _encode_prompt_learner(self.normal_prompt_learner, "normal")
        t_n = feats_n.mean(dim=0)  # [C]
        if self._normal_group_ids:
            t_n_components = _group_mean_by_ids(feats_n, self._normal_group_ids, self.component_count)
        else:
            t_n_components = _select_k(feats_n, self.component_count)
        t_n = self.g_t(t_n)
        if t_n_components is not None:
            t_n_components = self.g_t(t_n_components)
        t_n = F.normalize(t_n, dim=-1)
        if t_n_components is not None:
            t_n_components = F.normalize(t_n_components, dim=-1)

        # ---- Abnormal components ----
        feats_a = _encode_prompt_learner(self.abnormal_prompt_learner, "abnormal")
        if self._abnormal_group_ids:
            feats_a = _group_mean_by_ids(feats_a, self._abnormal_group_ids, self.abnormal_component_count)
        else:
            feats_a = _select_k(feats_a, self.abnormal_component_count)
        t_a = F.normalize(self.g_t(feats_a), dim=-1)

        return TextComponentBankOutput(t_n=t_n, t_a=t_a, t_n_components=t_n_components)

    def as_text_features(self) -> torch.Tensor:
        out = self()
        # [C, 2K] => K normal components + K abnormal components
        return torch.cat([out.t_n_components.T, out.t_a.T], dim=1)

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(list(self.abnormal_prompt_learner.parameters()))
        params.extend(list(self.normal_prompt_learner.parameters()))
        params.extend(self.text_encoder.trainable_parameters())
        params.extend(list(self.g_t.parameters()))
        return params

    def _load_llm_groups(self) -> None:
        if not self.llm_prompt_path or not os.path.isfile(self.llm_prompt_path):
            return
        with open(self.llm_prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        datasets = data.get("datasets", {})
        entry = datasets.get(self.dataset_key) or datasets.get(self.obj_name)
        if not entry:
            return
        self._llm_groups_normal = entry.get("normal", None)
        self._llm_groups_abnormal = entry.get("abnormal", None)
        self._llm_groups_background = entry.get("background", None)
        self._llm_groups_normal_list = self._normalize_groups(self._llm_groups_normal)
        self._llm_groups_abnormal_list = self._normalize_groups(self._llm_groups_abnormal)
        self._llm_groups_background_list = self._normalize_groups(self._llm_groups_background)
        self._llm_sentences_normal = self._flatten_sentences(self._llm_groups_normal)
        self._llm_sentences_abnormal = self._flatten_sentences(self._llm_groups_abnormal)
        self._llm_sentences_background = self._flatten_sentences(self._llm_groups_background)

    @staticmethod
    def _text_width(clip_model) -> int:
        if getattr(clip_model, "text_projection", None) is not None:
            proj = clip_model.text_projection
            if proj.ndim == 2:
                return int(proj.shape[1])
            return int(proj.shape[0])
        return int(clip_model.ln_final.weight.shape[0])

    @staticmethod
    def _flatten_sentences(groups: Optional[List]) -> List[str]:
        if not groups:
            return []
        out: List[str] = []
        for item in groups:
            if isinstance(item, list):
                out.extend([s for s in item if isinstance(s, str) and s.strip()])
            elif isinstance(item, str) and item.strip():
                out.append(item)
        # Keep insertion order while dropping duplicates
        return list(dict.fromkeys(out))

    @staticmethod
    def _normalize_groups(groups: Optional[List]) -> List[List[str]]:
        if not groups:
            return []
        out: List[List[str]] = []
        for item in groups:
            if isinstance(item, list):
                group = [s for s in item if isinstance(s, str) and s.strip()]
                if group:
                    out.append(group)
            elif isinstance(item, str) and item.strip():
                out.append([item])
        return out

    def _build_grouped_prompt_bank(
        self,
        *,
        grouped_sentences: List[List[str]],
        fallback_sentences: List[str],
        k: int,
    ) -> tuple[List[str], List[int]]:
        if k < 1:
            return [], []

        groups = [list(dict.fromkeys(g)) for g in grouped_sentences if len(g) > 0]
        if len(groups) == 0 and len(fallback_sentences) > 0:
            groups = [[s] for s in fallback_sentences if isinstance(s, str) and s.strip()]
        if len(groups) == 0:
            return [], []

        reps = (k + len(groups) - 1) // len(groups)
        groups = (groups * reps)[:k]

        prompts: List[str] = []
        group_ids: List[int] = []
        for gid, group in enumerate(groups):
            for sent in group:
                prompts.append(sent)
                group_ids.append(gid)
        return prompts, group_ids
