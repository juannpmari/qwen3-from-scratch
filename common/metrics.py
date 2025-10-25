import torch

class EvaluationMetrics:
    def compute_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size) CHECK
        labels: (batch_size, seq_len)
        """
        pass #TODO

    def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size) CHECK
        labels: (batch_size, seq_len)
        """
        pass #TODO