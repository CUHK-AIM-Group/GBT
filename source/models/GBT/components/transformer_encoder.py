from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import torch

def do_low_rank(weight, k, debug=False):

    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=1)

    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]

    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
        Inputs:
        query: (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        key: (S, N, E)(S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
        value: (S, N, E)(S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.
        key_padding_mask: (N, S)(N,S) , ByteTensor, where N is the batch size, S is the source sequence length.
        attn_mask: (L,S)(L,S) where L is the target sequence length, S is the source sequence length.
        
        Outputs:
        attn_output: (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        attn_output_weights: (N, L, S)(N,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length.
        '''
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)

        rate = 0.00001
        for i in range(len(weights)):
            weights[i] = do_low_rank(weights[i].type(torch.float32), \
                                     0.1 * rate)
        
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights