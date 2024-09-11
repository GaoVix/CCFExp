from torch import nn
import torch


class Attention_ID(nn.Module):
    def __init__(self, dim, in_chans, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, id, lm):

        B, N, C = id.shape  # [B, 49, 512]
        kv = self.kv(id).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)
        q = lm.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        h = self.proj(h)
        h = self.proj_drop(h)

        return h

class Attention_LM(nn.Module):
    def __init__(self, dim, in_chans, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, lm, em):

        B, N, C = lm.shape
        kv = self.kv(lm).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)
        q = em.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        h = self.proj(h)
        h = self.proj_drop(h)

        return h


class Attention_EM(nn.Module):
    def __init__(self, dim, in_chans, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, em, id):

        B, N, C = em.shape # [B, 49, 512]
        kv = self.kv(em).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)
        q = id.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        h = self.proj(h)
        h = self.proj_drop(h)

        return h

class Condition_Adapter(nn.Module):
    def __init__(self, dim, in_chans, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.att_id = Attention_ID(dim, in_chans, num_heads, qkv_bias, attn_drop, proj_drop)
        self.att_em = Attention_EM(dim, in_chans, num_heads, qkv_bias, attn_drop, proj_drop)
        self.att_lm = Attention_LM(dim, in_chans, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, id, em, lm):

        h_id = self.att_id(id, lm) # [B, 49, 512]
        h_em = self.att_em(em, id) # [B, 49, 512]
        h_lm = self.att_lm(lm, em) # [B, 49, 512]

        h = torch.cat([h_id, h_em, h_lm], dim=1).transpose(1, 2) # [B, 512, 147]

        return h
