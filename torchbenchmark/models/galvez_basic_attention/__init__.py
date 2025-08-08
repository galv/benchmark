import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(embed_dim, inner_dim)
        self.k_proj = nn.Linear(embed_dim, inner_dim)
        self.v_proj = nn.Linear(embed_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, t, _ = x.size()
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        print(x.shape, q.shape, k.shape, v.shape)
        # import ipdb; ipdb.set_trace()
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        attn = attn.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        x = x + self.dropout(self.out_proj(attn))
        x = self.norm1(x)
        ff = self.ffn(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x


class SmallTransformer(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, head_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.Sequential(
            TransformerBlock(embed_dim, num_heads, head_dim, dropout),
            # TransformerBlock(embed_dim, num_heads, head_dim, dropout),
        )
        self.cls = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        print("Initial shape=", x.shape)
        x = self.embed(x)
        x = self.layers(x)
        x = self.cls(x.mean(dim=1))
        return x


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 8
    DEFAULT_EVAL_BSIZE = 8

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        print("GALVEZ: batch_size=", batch_size)
        seq_len = 4096
        embed_dim = 1024
        num_heads = 4
        head_dim = 96
        dropout = 0.1
        self.model = SmallTransformer(seq_len, embed_dim, num_heads, head_dim, dropout).to(device)
        self.example_inputs = (torch.randint(0, seq_len, (self.batch_size, seq_len), device=device),)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def get_module(self):
        return self.model, self.example_inputs

    def forward(self):
        out = self.model(*self.example_inputs)
        target = torch.zeros_like(out)
        loss = self.criterion(out, target)
        return loss

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self):
        with torch.no_grad():
            out = self.model(*self.example_inputs)
        return (out,)
