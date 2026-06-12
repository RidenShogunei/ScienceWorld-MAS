"""Token-level clipped policy objective used by Main/Sub M-GRPO updates."""

from __future__ import annotations

import torch


def clipped_policy_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return PPO/GRPO clipped loss and detached diagnostics.

    Shapes are `[batch, tokens]`. One trajectory-level advantage is broadcast
    over its completion tokens. Padding and synchronized empty Sub slots must
    have `token_mask=0`.
    """
    if current_logprobs.shape != old_logprobs.shape or current_logprobs.shape != token_mask.shape:
        raise ValueError("log-probs and token_mask must have identical shapes")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    if advantages.ndim != 2 or advantages.shape[0] != current_logprobs.shape[0]:
        raise ValueError("advantages must have shape [batch] or [batch, 1]")
    if clip_low < 0 or clip_high < 0:
        raise ValueError("clip bounds must be non-negative")

    mask = token_mask.to(current_logprobs.dtype)
    denominator = mask.sum().clamp_min(1.0)
    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = clipped_ratio * advantages
    surrogate = torch.minimum(unclipped, clipped)
    loss = -(surrogate * mask).sum() / denominator

    with torch.no_grad():
        approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / denominator
        clip_fraction = ((ratio != clipped_ratio).to(mask.dtype) * mask).sum() / denominator
    return loss, {
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "mean_ratio": ((ratio * mask).sum() / denominator).detach(),
        "active_tokens": mask.sum().detach(),
    }


def add_reference_kl(
    policy_loss: torch.Tensor,
    current_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a non-negative sampled-token KL penalty against the SFT reference."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if current_logprobs.shape != reference_logprobs.shape:
        raise ValueError("policy and reference log-probs must align")
    mask = token_mask.to(current_logprobs.dtype)
    denominator = mask.sum().clamp_min(1.0)
    reference_minus_policy = reference_logprobs - current_logprobs
    per_token_kl = torch.exp(reference_minus_policy) - reference_minus_policy - 1.0
    sampled_kl = (per_token_kl * mask).sum() / denominator
    return policy_loss + beta * sampled_kl, sampled_kl.detach()
