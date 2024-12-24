import torch
import math
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """
    modified from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # the length of position encoding x the dimension of the model
        # 0::2 means each even column
        # 1::2 means each odd column
        pe[:, 0::2] = torch.sin(position * div_term)
        # if d_model is even
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:  # if d_model is odd, remove the last column of div_term
            pe[:, 1::2] = torch.sin(position * div_term)[:, :-1]
        self.register_buffer('pe', pe)

    def forward(self) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # x = x + self.pe[:x.size(0)]
        return self.dropout(self.pe.T)
