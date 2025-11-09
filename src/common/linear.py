import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device  = None,
        dtype: torch.dtype  = None,
    ):
        """
        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            device (torch.device , optional): device to run on. Defaults to None.
            dtype (torch.dtype , optional): data type to run on. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        init_std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=init_std, a=-init_std * 3.0, b=init_std * 3.0
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x (torch.tensor): input tensor
        Returns:
            torch.tensor: output tensor
        """
        return x.matmul(self.weight.t())
