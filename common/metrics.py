import torch

def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, context_length, vocab_size) CHECK
    targets: (batch_size, context_length) IDs of the target tokens

    Returns:
        ce: (batch_size, context_length) cross entropy for each token of each sequence in the batch
    """
    ce = 0
    for b in range(logits.shape[0]): #D
        for i in range(targets.shape[1]): #i
            target = targets[b][i]
            prob = torch.exp(logits[b][i][target])/sum(torch.exp(logits[b][i]))
            ce += -torch.log(prob)
    ce = ce / (logits.shape[0] * logits.shape[1])
    return ce # (batch_size, context_length)

def compute_perplexity(cross_entropy: torch.Tensor) -> torch.Tensor:
    """
    cross_entropy: (batch_size, context_length)

    Returns:
        perplexity: (batch_size)
    """
    return torch.exp(torch.mean(cross_entropy, dim=1)) # (batch_size)