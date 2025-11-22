import torch


class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q of shape (batch_size, seq_len_q, dim)
        K of shape (batch_size, seq_len_k, dim)
        V of shape (batch_size, seq_len_k, dim)
        """

        bq = 16
        bk = 16

        batch_size = Q.size(0)
        Nq = Q.size(1)
        Nk = K.size(1)
        d = Q.size(2)

        Tq = Nq // bq
        Tk = Nk // bk

        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, Nq, 1)
        k_tiles = torch.split(K, bk, dim=1)
        v_tiles = torch.split(V, bk, dim=1)

        for i in range(Tq):
            Qi = Q[:, i * bq : (i + 1) * bq, :]
            Oi_new = torch.zeros_like(Qi)
            li = torch.zeros(batch_size, bq, 1, device=Q.device)
            mi = torch.full((batch_size, bq, 1), float("-inf"), device=Q.device)

            for j in range(Tk):
                Kj = k_tiles[j]  # (batch_size, bk, d)
                Vj = v_tiles[j]  # (batch_size, bk, d)

                Sij = Qi @ Kj.transpose(-2, -1) / (d**0.5)  # Sij: (batch_size, bq, bk)

                mi_new = torch.max(
                    mi, torch.max(Sij, dim=-1, keepdim=True).values
                )  # mi_new: (batch_size, bq, 1)
                mi_scale = torch.exp(mi - mi_new)
                Pij = torch.exp(Sij - mi_new)  # (batch_size, bq, bk)
                li_new = mi_scale * li + torch.sum(Pij, dim=-1, keepdim=True)
                Oi_new = mi_scale * Oi_new + Pij @ Vj

                mi = mi_new
                li = li_new

            Li = mi + torch.log(li)  # (batch_size, bq, 1)
            Oi_final = Oi_new / li

            O[:, i * bq : (i + 1) * bq, :] = Oi_final  # (batch_size, bq, d)
            L[:, i * bq : (i + 1) * bq, :] = Li  # (batch_size, bq, 1)

        ctx.save_for_backward(Q, K, V, O, L.squeeze(-1))
        return O, L

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented yet.")


# uv run pytest -k test_flash_forward_pass_pytorch
