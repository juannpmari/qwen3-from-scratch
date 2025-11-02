import torch

def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, vocab_size)
    targets: (batch_size,) IDs of the target tokens

    Returns:
        ce: scalar; avg cross entropy for the sequences in the batch
    """
    ce = torch.zeros(logits.shape[0])
    for b in range(logits.shape[0]): #iterate over batch
        sequence = logits[b] #vocab_size
        sequence_log_prob = torch.log_softmax(sequence, dim=0)
        target_log_prob = sequence_log_prob[targets[b]]
        ce[b] = -target_log_prob
    return ce.mean()  #scalar


    #uv run pytest -k test_cross_entropy