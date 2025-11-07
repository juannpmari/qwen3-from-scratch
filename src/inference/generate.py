from src.common.softmax import softmax
import torch
from torch import nn
from src.preprocessing.tokenizer import BPETokenizer

def _sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling from a probability distribution.

    Args:
        probs (torch.Tensor): shape (vocab_size,) or (batch_size, vocab_size)
        top_p (float): cumulative probability threshold (e.g. 0.9)

    Returns:
        torch.Tensor: sampled token IDs, same batch size as input (or scalar if 1D)
    """
    probs = probs[:,-1,:]
    if probs.dim() not in (1, 2):
        raise RuntimeError("probs must be 1 or 2 dim")

    single = False
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)  # -> (1, vocab)
        single = True

    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # (B, V)

    # Cumulative sum over sorted probs
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, V)

    # Keep tokens with cumulative_probs <= top_p. Always keep the first token.
    keep_mask = cumulative_probs <= top_p
    keep_mask[..., 0] = True  # ensure at least the top token is kept

    # Zero out probs beyond top_p
    filtered_probs = sorted_probs * keep_mask.to(sorted_probs.dtype)  # (B, V)

    # Compute sums per row
    row_sums = filtered_probs.sum(dim=-1, keepdim=True)  # (B, 1)

    # For rows where row_sums == 0 (shouldn't normally happen because we kept [0]), fallback:
    zero_sum_mask = (row_sums == 0).squeeze(-1)  # (B,)
    if zero_sum_mask.any():
        # Set filtered_probs to contain only the top token in those rows
        # top token is at position 0 in sorted_probs
        filtered_probs[zero_sum_mask] = 0.0
        filtered_probs[zero_sum_mask, 0] = 1.0
        row_sums = filtered_probs.sum(dim=-1, keepdim=True)

    # Renormalize
    normalized = filtered_probs / row_sums

    # Now sample (torch.multinomial accepts 2D or 1D)
    sampled_in_sorted = torch.multinomial(normalized, num_samples=1)  # shape (B, 1)

    # Gather original token indices
    sampled_tokens = sorted_indices.gather(-1, sampled_in_sorted)  # (B, 1)
    sampled_tokens = sampled_tokens.squeeze(-1)  # (B,)

    if single:
        return sampled_tokens.squeeze(0)  # return scalar tensor
    return sampled_tokens  # (batch_size,)

def generate_text(model:nn.Module, tokenizer:BPETokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float):
    device = next(model.parameters()).device
    generated_tokens = 0
    end_of_sequence = False
    token_ids = tokenizer.encode(prompt)
    
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    while generated_tokens < max_tokens and not end_of_sequence:
        logits = model(token_ids)
        probs = softmax(logits, dim=-1, temperature=temperature)

        token_id = _sample(probs, top_p=top_p)
        generated_tokens += 1
        token_id = token_id.unsqueeze(0)  # [1] → [1, 1]
        token_ids = torch.cat([token_ids, token_id], dim=1)
        # token_ids = torch.cat([token_ids, token_id.unsqueeze(0)], dim=-1)


    response = ''.join([char for char in tokenizer.decode(token_ids[0].tolist())])

    return response
        