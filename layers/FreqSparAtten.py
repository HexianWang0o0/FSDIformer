import numpy as np
import torch.fft
import torch
import torch.nn as nn
from math import sqrt

    
def get_random_frequencies(seq_len, selected_freq_count=64): 

    n = min(selected_freq_count, seq_len//2)
    index = list(range(0, seq_len // 2))    
    np.random.shuffle(index)
    index = index[:n]       
    index.sort()  

    return index


class FreqSparAtten(nn.Module):

    def __init__(self, seq_len_q, seq_len_kv, selected_freq_count=32):      
        super(FreqSparAtten, self).__init__()
        
        self.index_q = get_random_frequencies(seq_len_q, selected_freq_count)
        self.index_kv = get_random_frequencies(seq_len_kv, selected_freq_count)

    def forward(self, q, k, v, attn_mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape    
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)  
        xv = v.permute(0, 2, 3, 1)  

        scale = 1. / sqrt(E)
        
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device)   
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq[:, :, :, j]     

        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device)     
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk[:, :, :, j]      

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))        
        xqk_ft = torch.softmax(scale*xqk_ft, dim=-1)                          
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)                    
        
        out_ft = torch.zeros(B, H, E, L, device=xq.device)       
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkv_ft[:, :, :, i]      
        
        return (out_ft.contiguous(), xqk_ft) 
    

class FreqSparAttn_Layer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, scale=None, **_):
        super(FreqSparAttn_Layer, self).__init__()
        
        self.scale = scale

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=2, stride=2)
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=2, stride=2)
        self.value_projection = nn.Conv1d(d_model, d_values * n_heads, kernel_size=2, stride=2)
        self.out_projection = nn.Linear(d_values * n_heads, d_model * 2)
        self.n_heads = n_heads
        self.attn = attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries_fft = torch.fft.rfft(queries.permute(0, 2, 1))
        keys_fft = torch.fft.rfft(keys.permute(0, 2, 1))

        L_fft = queries_fft.shape[-1]
        S_fft = keys_fft.shape[-1]
        
        queries_fft = torch.stack([queries_fft.real, queries_fft.imag], -1)
        queries_fft = queries_fft.reshape(B, queries_fft.shape[1], -1)
        queries_fft = self.query_projection(queries_fft).permute(0, 2, 1).view(B, L_fft, H, -1)        
        
        keys_fft = torch.stack([keys_fft.real, keys_fft.imag], -1)
        keys_fft = keys_fft.reshape(B, keys_fft.shape[1], -1)
        keys_fft = self.key_projection(keys_fft).permute(0, 2, 1).view(B, S_fft, H, -1)

        V, attn = self.attn(queries_fft, keys_fft, keys_fft, attn_mask=attn_mask)                 
        V = V.permute(0, 3, 1, 2)   
        V = V.contiguous().view(B, L_fft, -1)       

        V = self.out_projection(V)
        V = V.view(B, L_fft, -1, 2)

        V = torch.complex(V[..., 0], V[..., 1]).permute(0, 2, 1)

        V = torch.fft.irfft(V, n=L).permute(0, 2, 1)

        return V.contiguous(), attn
