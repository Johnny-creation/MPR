import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from typing import Optional

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss


class LossSigmoid(nn.Module):
    def __init__(self ,dec_type ="mean", lr=0.01):
        super(LossSigmoid, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, logits, labells, dec):
        # Transform label to [-1, 1]
        labels = (2.0 * labells - 1.0).unsqueeze(-1)  # Dim: [batch_size, 1]
        # Create label_all
        label_all = torch.cat([-1 * labels, labels], dim=1)  # Dim: [batch_size, 2]. [:, 0]normality--- [:,1] abnormality
        input_sig = label_all.unsqueeze(1) * logits # [batch_size, 289, 2]
        loss_each = self.log_sigmoid(input_sig) # [batch_size, 289, 2]
        if torch.isnan(loss_each).any():
            print("NaN in loss")
        # mean or max or combination for decision
        dec_each_in = dec(loss_each)  # [batch_size, 2]

        loss = - torch.mean(torch.sum(dec_each_in, dim=-1))
        if torch.isnan(loss).any():
            print("loss")
        if torch.isnan(dec_each_in).any():
            print("dec_each_in")
            print("loss_each")
        return loss


    def validation(self, logits, dec):
        anomaly_score = F.softmax(logits, dim=-1)
        anomaly_score = dec(anomaly_score[:,:,1])
        return anomaly_score


class LossSoftmaxBased(nn.Module):
    def __init__(self, dec_type="mean"):
        super(LossSoftmaxBased, self).__init__()
        self.loss_bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, dec):
        logits = F.softmax(logits, dim=-1)  # [batch size, :,0] is normality score , [batch size, :,1] is anomaly score, the shape should be [batch size,289,2]
        normality_score = dec(logits[:,:,0])
        anomaly_score = dec(logits[:,:,1])
        loss = 0
        loss += self.loss_bce(1 - normality_score, labels)
        loss += self.loss_bce(anomaly_score, labels)
        return loss

    def validation(self, logits, dec):
        logits = F.softmax(logits, dim=-1)
        anomaly_score = dec(logits[:,:,1])
        return anomaly_score


class Loss_detection(nn.Module):
    def __init__(self,args, device, loss_type="sigmoid", dec_type="mean", lr=0.001):
        super(Loss_detection, self).__init__()
        self.img_size = args.img_size
        self.notuseful = nn.Parameter(torch.zeros(1, device=device))
        self.log_sigmoid = nn.LogSigmoid()
        if dec_type == "mean":
            self.decision = lambda a: (torch.mean(a, dim=1))
        elif dec_type == "max":
            self.decision = lambda a: (torch.max(a, dim=1)[0])
        elif dec_type == "both":
            self.alphadec = (torch.ones(1) * 0.0).to(device)  # Initialize with 0.5
            self.decision = lambda a: (torch.sigmoid(self.alphadec) * torch.mean(a, dim=1) +
                                       (1 - torch.sigmoid(self.alphadec)) * torch.max(a, dim=1)[0])
        self.loss_type = loss_type
        if loss_type == "softmax":
            self.loss_softmax = LossSoftmaxBased(dec_type)
        elif loss_type == "sigmoid":
            self.loss_sigmoid = LossSigmoid(dec_type=dec_type)
            print("sigmoid loss in the house")
        else:
            print("not implemented")
            exit(10)
        self.to(device)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, logits, labels):
        det_loss_final = 0
        if self.loss_type == "softmax":
            det_loss_final+= self.loss_softmax(logits, labels, dec= self.decision)
        elif self.loss_type == "sigmoid":
            det_loss_final += self.loss_sigmoid(logits, labels, dec= self.decision)
        elif self.loss_type == "both":
            ranged_alpha = torch.sigmoid(self.alphaloss)
            det_loss_final+= ( ranged_alpha * self.loss_softmax(logits, labels, dec= self.decision)
                              + (1.0- ranged_alpha) * self.loss_sigmoid(logits, labels, dec= self.decision))

        return det_loss_final

    def validation(self, logits):
        if self.loss_type == "softmax":
            return self.loss_softmax.validation(logits, dec=self.decision)
        elif self.loss_type == "sigmoid":
            return self.loss_sigmoid.validation(logits, dec=self.decision)
        elif self.loss_type == "both":
            ranged_alpha = torch.sigmoid(self.alphaloss)
            anomaly_score = (ranged_alpha * self.loss_softmax.validation(logits, dec=self.decision)
                         + (1.0 - ranged_alpha) * self.loss_sigmoid.validation(logits, dec=self.decision))
            return anomaly_score

    def sync_AS(self, logits):
        B, L, C = logits.shape
        H = int(np.sqrt(L))
        logits = F.interpolate(logits.permute(0, 2, 1).view(B, C, H, H),
                               size=self.img_size, mode='bilinear', align_corners=True)

        if self.loss_type == "softmax":
            logits = torch.softmax(logits, dim=1)
        elif self.loss_type == "sigmoid":
            logits = torch.softmax(logits, dim=1)
        elif self.loss_type == "both":
            ranged_alpha = torch.sigmoid(self.alphaloss)
            logits = (ranged_alpha * torch.softmax(logits, dim=1)) + (1.0 - ranged_alpha) * F.sigmoid(logits)
        return logits


def _margin_from_logits(logits: torch.Tensor, contrast_mood: str) -> torch.Tensor:
    """
    Args:
        logits: [..., 2] where logits[..., 0] is normal, logits[..., 1] is abnormal
        contrast_mood: 'yes' or 'no'

    Returns:
        margin m = sim_a - sim_n with shape logits.shape[:-1]
    """
    if logits.size(-1) != 2:
        raise ValueError(f"Expected last dim == 2, got {tuple(logits.shape)}")
    if contrast_mood == "yes":
        # In contrast mode, abnormal branch logit is (sim_a - sim_n).
        return logits[..., 1]
    if contrast_mood == "no":
        return logits[..., 1] - logits[..., 0]
    raise ValueError(f"Invalid contrast_mood={contrast_mood}, expected 'yes' or 'no'")

class ComponentLosses(nn.Module):
    """
    Adds:
      - L_diff: token/image margin loss (supports optional pixel mask)
      - L_div: prototype diversity loss (cosine hinge over component pairs)
    """

    def __init__(self, args):
        super().__init__()
        self.img_size = int(getattr(args, "img_size", 240))
        self.diff_margin = float(getattr(args, "diff_margin", 0.2))
        self.lam_diff = float(getattr(args, "lam_diff", 0.0))
        self.lam_div = float(getattr(args, "lam_div", 0.0))
        self.div_margin = float(getattr(args, "div_margin", 0.4))
        self.contrast_mood = getattr(args, "contrast_mood", "yes")
        self.component_count = int(getattr(args, "component_count", 6))
        self.optimizer = None

    def forward(
        self,
        *,
        det_logits_list,
        seg_logits_list,
        labels: torch.Tensor,
        text_features: torch.Tensor,
        mask: "Optional[torch.Tensor]" = None,
    ):
        # Text regularizer
        t_n, t_a = self._split_text_features(text_features)
        t_n = F.normalize(t_n, dim=-1)
        t_a = F.normalize(t_a, dim=-1)

        l_diff = t_n.sum() * 0.0
        if self.lam_diff != 0:
            if mask is not None and seg_logits_list is not None and len(seg_logits_list) > 0:
                l_diff = self._diff_with_mask(seg_logits_list, mask)
            else:
                l_diff = self._diff_image(det_logits_list, labels)
        l_div = t_n.sum() * 0.0
        if self.lam_div != 0:
            l_div = self._prototype_diversity_loss(t_n, t_a)

        diff_term = self.lam_diff * l_diff
        div_term = self.lam_div * l_div
        total = diff_term + div_term
        w_diff = diff_term.new_tensor(float(self.lam_diff))
        w_div = diff_term.new_tensor(float(self.lam_div))

        return total, {
            "L_diff": l_diff,
            "L_div": l_div,
            "W_diff": w_diff,
            "W_div": w_div,
        }

    def _prototype_diversity_loss(self, t_n: torch.Tensor, t_a: torch.Tensor) -> torch.Tensor:
        loss_n = self._pair_div_hinge(t_n)
        loss_a = self._pair_div_hinge(t_a)
        return loss_n + loss_a

    def _pair_div_hinge(self, p: torch.Tensor) -> torch.Tensor:
        # p: [K, C]
        k = int(p.size(0))
        if k <= 1:
            return p.sum() * 0.0

        p_hat = F.normalize(p, dim=-1, eps=1e-6)
        sim = torch.matmul(p_hat, p_hat.transpose(0, 1))  # [K, K]
        off_diag = ~torch.eye(k, device=sim.device, dtype=torch.bool)
        sim_pairs = sim[off_diag]
        penalties = F.relu(sim_pairs - self.div_margin).pow(2)
        return penalties.mean()

    def _split_text_features(self, text_features: torch.Tensor):
        if text_features.ndim != 2:
            raise ValueError(
                f"text_features must be [C,2K], got {tuple(text_features.shape)}"
            )
        expected = 2 * self.component_count
        if text_features.size(1) != expected:
            raise ValueError(
                f"text_features second dim must be exactly 2K={expected}, got {tuple(text_features.shape)}"
            )
        t_n = text_features[:, : self.component_count].T  # [K, C]
        t_a = text_features[:, self.component_count : expected].T  # [K, C]
        return t_n, t_a

    def _diff_image(self, det_logits_list, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B] with {0,1}
        margins = []
        for logits in det_logits_list:
            m = _margin_from_logits(logits, self.contrast_mood)  # [B, N]
            margins.append(m.mean(dim=1))  # [B]
        m_img = torch.stack(margins, dim=0).mean(dim=0) if len(margins) else labels.float() * 0.0

        labels = labels.float().view(-1)
        anom = labels > 0.5
        norm = ~anom

        loss = m_img.sum() * 0.0
        if anom.any():
            loss = loss + F.softplus(self.diff_margin - m_img[anom]).mean()
        if norm.any():
            loss = loss + F.softplus(self.diff_margin + m_img[norm]).mean()
        return loss

    def _diff_with_mask(self, seg_logits_list, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B,1,H,W] or [B,H,W] or [H,W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5)

        losses = []
        for logits in seg_logits_list:
            m = _margin_from_logits(logits, self.contrast_mood)  # [B, L]
            B, L = m.shape
            H = int(np.sqrt(L))
            if H * H != L:
                continue
            m_map = m.view(B, 1, H, H)
            m_map = F.interpolate(m_map, size=self.img_size, mode="bilinear", align_corners=True)

            mask_b = mask
            if mask_b.size(0) != B:
                mask_b = mask_b.expand(B, -1, -1, -1)
            anom_vals = m_map[mask_b]
            norm_vals = m_map[~mask_b]

            loss = m_map.sum() * 0.0
            if anom_vals.numel() > 0:
                loss = loss + F.softplus(self.diff_margin - anom_vals).mean()
            if norm_vals.numel() > 0:
                loss = loss + F.softplus(self.diff_margin + norm_vals).mean()
            losses.append(loss)

        if len(losses) == 0:
            return mask.float().sum() * 0.0
        return torch.stack(losses).mean()

"""
    def sync_modality(self, normal_modall, abnormal_modall, cur_epoch, cur_modal_label, branch_type):
        loss = 0
        cur_modal_label_ranged = cur_modal_label + 2


        normal_modal = normal_modall * self.temp.get_value(cur_epoch) + self.bias  # [batch size, 289, 2]
        normal_modal = F.softmax(normal_modal, dim=-1)
        final_score_modal_normal = torch.mean(normal_modal, dim=1)
        loss += self.ce_loss(final_score_modal_normal, cur_modal_label_ranged)


        if branch_type == "2branch":
            abnormal_modal = abnormal_modall * self.temp.get_value(cur_epoch) + self.bias  # [batch size, 289, 2]
            abnormal_modal = F.softmax(abnormal_modal, dim=-1)
            final_score_modal_abnormal = torch.mean(abnormal_modal, dim=1)
            loss += self.ce_loss(final_score_modal_abnormal, cur_modal_label_ranged)


        return loss
"""
