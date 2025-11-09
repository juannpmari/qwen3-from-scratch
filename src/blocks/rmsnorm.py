import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization layer
    """

    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            head_dim (int): head dimension
            eps (float, optional): epsilon value. Defaults to 1e-5.
            device (torch.device | None, optional): device to run on. Defaults to None.
            dtype (torch.dtype | None, optional): data type to run on. Defaults to None.
        """
        super().__init__()
        self.epsilon = eps
        self.gain = torch.nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x (torch.tensor): input tensor
        Returns:
            torch.tensor: output tensor
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # To avoid overflow when squaring
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        result = self.gain * x / rms
        return result.to(in_dtype)
