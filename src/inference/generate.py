from src.common.softmax import softmax
import torch
from torch import nn
from src.preprocessing.tokenizer import BPETokenizer

def _sample(probs: torch.Tensor, top_p: float) -> int:
    pass

def generate(model:nn.Module, tokenizer:BPETokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float):
    device = next(model.parameters()).device
    generated_tokens = 0
    end_of_sequence = False
    
    while generated_tokens < max_tokens and not end_of_sequence:
        token_ids = tokenizer.encode(prompt)
        token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
        logits = model(token_ids)
        probs = softmax(logits, dim=-1, temperature=temperature)

        token_id = _sample(probs, top_p=top_p)
        generated_tokens += 1
        prompt += tokenizer.decode(token_id)

    return prompt
        