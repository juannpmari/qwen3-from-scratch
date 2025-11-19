import torch


class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        """
        Q of shape (batch_size, seq_len, dim)
        K of shape ()
        V of shape 
        """
        bq = 16 # tile size for Q
        bk = 16 # tile size for K, V

        d = Q.size(-1)
        Tq = Q.size(-2)/bq
        Tk = K.size(-2)/bk
        # ctx.save_for_backward(Q, K, V, O, L)  # Save tensors for backward pass if needed

        O = torch.zeros_like(Q)  # Output tensor
        L = torch.zeros(Q.size(0), 1)  # Log-sum-exp tensor

        #Split Q
        q_tiles = torch.split(Q, Tq, dim=0) # List of Q tiles, each one bq x d #TODO: check border conditions
        
        # Split K, V
        k_tiles = torch.split(K, Tk, dim=0) # List of K tiles, each one bk x d
        v_tiles = torch.split(V, Tk, dim=0) # List of V tiles, each one bk x d

        for i in range(0,Tq):
            Qi = q_tiles[i] # Load from HBM to SRAM
            Oi = torch.zeros(bq, d)
            li = torch.zeros(bq, 1)
            mi = torch.full((bq, 1), float('-inf'))

            for j in range(0,Tk):
                Kj = k_tiles[j] # Load from HBM to SRAM
                Vj = v_tiles[j] # Load from HBM to SRAM

                Sij = Qi @ Kj.transpose(-2, -1) / (d ** 0.5)  # bq x bk
                if j > 0:
                    mi[j] = torch.max(mi[j-1], torch.max(Sij, dim=-1, keepdim=True).values)  # bq x 1
                else:
                    mi[j] = torch.max(Sij, dim=-1, keepdim=True).values  # bq x 1
                
                Pij = torch.exp(Sij - mi[j])  # bq x bk

                lij = torch.exp(mi[j-1] - mi[j]) * li[j-1] + torch.sum(Pij, dim=-1, keepdim=True)
                Oij = None # TBD  # bq x d

            Oi = None # TBD
            Li = None # TBD
            O[i * bq:(i + 1) * bq, :] = Oi  # Store back to HBM
            L[i * bq:(i + 1) * bq, :] = Li  # Store back to HBM
        return O, L






    

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented yet.")
    
# uv run pytest -k test_flash_forward_pass_pytorch