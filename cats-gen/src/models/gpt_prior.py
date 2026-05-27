import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.n_heads   = n_heads
        self.head_dim  = hidden_dim // n_heads

        self.qkv  = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        # causal mask via scaled dot-product attention (PyTorch >= 2.0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))


class GPTBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTPrior(nn.Module):
    """
    Autoregressive Transformer prior for VQ-VAE.

    Flattens the (H, W) latent grid into a sequence of length H*W,
    predicts the next codebook index at each position.
    """

    def __init__(self, num_embeddings, latent_h, latent_w,
                 hidden_dim=256, n_layers=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.K        = num_embeddings
        self.H        = latent_h
        self.W        = latent_w
        self.seq_len  = latent_h * latent_w

        self.tok_emb = nn.Embedding(num_embeddings, hidden_dim)
        self.pos_emb = nn.Embedding(self.seq_len, hidden_dim)
        
        self.sos_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.drop    = nn.Dropout(dropout)

        self.blocks  = nn.Sequential(*[
            GPTBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ln_out  = nn.LayerNorm(hidden_dim)
        self.head    = nn.Linear(hidden_dim, num_embeddings, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, indices):
        """
        indices: (B, H, W)
        returns logits: (B, K, H, W)  — same shape convention as PixelCNN
        """
        B = indices.shape[0]
        seq = indices.view(B, self.seq_len)                          # (B, T)

        token_embeddings = self.tok_emb(seq)                         # (B, T, C)
        
        sos_expanded = self.sos_token.expand(B, 1, -1)               # (B, 1, C)
        input_embeddings = torch.cat([sos_expanded, token_embeddings[:, :-1, :]], dim=1) # (B, T, C)

        pos = torch.arange(self.seq_len, device=indices.device)
        x   = self.drop(input_embeddings + self.pos_emb(pos))        # (B, T, C)

        x   = self.blocks(x)
        x   = self.ln_out(x)
        logits = self.head(x)                                        # (B, T, K)

        # reshape to (B, K, H, W) to match cross_entropy expectation
        logits = logits.view(B, self.H, self.W, self.K)
        logits = logits.permute(0, 3, 1, 2).contiguous()            # (B, K, H, W)
        return logits

    @torch.no_grad()
    def generate(self, shape, device, temperature=1.0):
        """
        Autoregressively sample a grid of indices.
        shape: (B, H, W)
        """
        B, H, W = shape
        seq_len = H * W
        seq = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        self.eval()
        for t in range(seq_len):
            indices_2d = seq.view(B, H, W)
            logits = self.forward(indices_2d)                       # (B, K, H, W)
            
            logits_flat = logits.view(B, self.K, seq_len)           # (B, K, T)
            logits_t = logits_flat[:, :, t] / temperature
            
            top_k = 30
            v, _ = torch.topk(logits_t, top_k)
            logits_t[logits_t < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits_t, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            seq[:, t] = sampled

        return seq.view(B, H, W)