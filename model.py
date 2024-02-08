from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import encode_decode
from dataclasses import dataclass
import pickle
from pathlib import Path
@dataclass
class ModelConfig:
    jp_vocab_size: int
    en_vocab_size: int
    device: torch.device
    block_size:int = encode_decode.DEFAULT_BLOCK_SIZE
    n_emb: int = 256
    n_heads: int = 8
    n_layer: int = 3
    dropout: float = .1
    ff_scale: int = 4
    @property
    def head_size(self) -> int:
        assert self.n_emb % self.n_heads == 0, f"n_heads should divide n_emb evenly, found {self.n_emb}%{self.n_heads} = {self.n_emb % self.n_heads}"
        return self.n_emb // self.n_heads
    def write(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def tryload(cls, filename: str) -> Optional['ModelConfig']:
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except OSError:
            return None
 

class Head(nn.Module):
    def __init__(self, config: ModelConfig, masked=True, ):
        super().__init__()
        self.query = nn.Linear(config.n_emb, config.head_size, bias=False)
        self.key = nn.Linear(config.n_emb, config.head_size, bias=False)
        self.value = nn.Linear(config.n_emb, config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
        self.masked = masked

        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        weight = query @ key.transpose(-2, -1) * C ** -.5

        if self.masked:
            weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        return weight @ value

class CrossHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.query = nn.Linear(config.n_emb, config.head_size, bias=False)
        self.key = nn.Linear(config.n_emb, config.head_size, bias=False)
        self.value = nn.Linear(config.n_emb, config.head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x_dec, x_enc):
        _, _, C = x_enc.shape
        query = self.query(x_dec)
        key = self.key(x_enc)
        value = self.value(x_enc)
        weight = query @ key.transpose(-2, -1) * C ** -.5

        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        return weight @ value

class MultiHead(nn.Module):
    def __init__(self, config: ModelConfig, masked=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, masked) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MultiCrossHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([CrossHead(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_emb,config. n_emb)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x_dec, x_enc):
        out = torch.cat([h(x_dec, x_enc) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FFBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_emb, config.ff_scale*config.n_emb),
            nn.ReLU(),
            nn.Linear(config.ff_scale*config.n_emb, config.n_emb),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.network(x)

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.att = MultiHead(config)
        self.cross_attn = MultiCrossHead(config)
        self.ff = FFBlock(config)
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)
        self.ln3 = nn.LayerNorm(config.n_emb)
        
    def forward(self, x, x_enc):
        x = x + self.att(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), x_enc)
        x = x + self.ff(self.ln3(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.att = MultiHead(config, masked=False)
        self.ff = FFBlock(config)
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)
        
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
        
class DecoderStack(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])

    def forward(self, x_dec, x_enc):
        for layer in self.layers:
            x_dec = layer(x_dec, x_enc)
        return x_dec
            
class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.enc_token_embedding_table = nn.Embedding(config.jp_vocab_size, config.n_emb)
        self.enc_position_embedding_table = nn.Embedding(config.block_size, config.n_emb)
        self.dec_token_embedding_table = nn.Embedding(config.en_vocab_size, config.n_emb)
        self.dec_position_embedding_table = nn.Embedding(config.block_size, config.n_emb)
        self.decoder_blocks = DecoderStack(config)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_emb) # final layer norm
        self.lm_head = nn.Linear(config.n_emb, config.en_vocab_size)
        self.config = config

    def forward(self, enc_idx, dec_idx, targets=None):
        B, T_enc = enc_idx.shape
        _, T_dec = dec_idx.shape

        # both (B,T_enc) tensor of integers
        enc_tok_emb = self.enc_token_embedding_table(enc_idx) # (B,T_enc,C)
        enc_pos_emb = self.enc_position_embedding_table(torch.arange(T_enc, device=self.config.device)) # (T_enc,C)

        x_enc = enc_tok_emb + enc_pos_emb

        y_enc = self.encoder_blocks(x_enc)

        dec_tok_emb = self.dec_token_embedding_table(dec_idx) # (B,T_dec,C)
        dec_pos_emb = self.dec_position_embedding_table(torch.arange(T_dec, device=self.config.device)) # (T_dec,C)

        x_dec = dec_tok_emb + dec_pos_emb # (B,T_dec,C)
        # HERE: x_enc -> y_enc
        x = self.decoder_blocks.forward(x_dec, y_enc) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=encode_decode.PAD_I)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
