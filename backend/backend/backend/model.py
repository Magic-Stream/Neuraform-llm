"""
🧠 NEURAFORM Model Definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
#                    KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = """
Physics is the fundamental science that studies matter, energy, space, and time.
The universe is governed by four fundamental forces: gravity, electromagnetism, strong nuclear force, and weak nuclear force.
Albert Einstein developed the theory of relativity, which revolutionized our understanding of space, time, and gravity.
The equation E equals mc squared shows that energy and mass are equivalent and interchangeable.
Isaac Newton discovered the laws of motion and universal gravitation.
Quantum mechanics describes the behavior of particles at the atomic and subatomic level.
The speed of light in a vacuum is approximately 299,792,458 meters per second.
Chemistry is the science of matter and the changes it undergoes.
The periodic table organizes all known chemical elements by atomic number.
Biology is the study of living organisms and life processes.
The cell is the basic unit of life.
DNA is the molecule that carries genetic information.
Evolution is the change in inherited characteristics over generations.
Charles Darwin proposed the theory of natural selection.
Mathematics is the study of numbers, quantities, and shapes.
Computer science is the study of computation and information processing.
Artificial intelligence creates systems that can learn and reason.
Machine learning trains computers to learn from data.
Neural networks are inspired by biological brains.
Deep learning uses neural networks with many layers.
The Internet is a global network of networks.
History is the study of past events and human civilization.
Geography is the study of Earth's landscapes, environments, and places.
Literature is written works of artistic merit.
Philosophy is the study of fundamental questions about existence, knowledge, and ethics.
Psychology is the scientific study of mind and behavior.
Economics studies how societies allocate scarce resources.
Art is the expression of human creativity and imagination.
Music is the art of organized sound.
Technology applies scientific knowledge for practical purposes.
Medicine is the science of diagnosing, treating, and preventing disease.
The environment includes all living and non-living things on Earth.
Space exploration has sent probes throughout the solar system.
The universe contains all matter, energy, space, and time.
""" * 20  # Repeat for more training data


# ============================================================
#                    TOKENIZER
# ============================================================

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        return [self.char2idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.idx2char.get(t, '') for t in tokens])


# ============================================================
#                    MODEL COMPONENTS
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)
    
    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.rotary = RotaryEmbedding(self.head_dim, block_size * 2)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), 1).bool())
        
        self.flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=0.1 if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        hidden = int(n_embd * 4 * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64
        
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
#                    NEURAFORM MODEL
# ============================================================

class Neuraform(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=8, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) 
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"🧠 Neuraform: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        x = self.tok_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
        self.eval()
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                indices_to_remove = mask.scatter(1, sorted_idx, mask)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
