import torch
import torch.nn as nn
from layers.Functionality import MLPLayer


class EncoderLayer(nn.Module):
    def __init__(self, freqspar_atten, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.freqspar_atten = freqspar_atten
        self.feedforward_freq = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout,activation=activation)  
        self.norm_freq = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, **kwargs):
        # Frequency Sparse Attention
        x_freq = x
        freq_out, attn_freq = self.freqspar_atten(x_freq, x_freq, x_freq, attn_mask=attn_mask)

        # Add and Norm
        x_freq = self.norm_freq(x_freq + self.dropout(freq_out))        

        # Feedforward
        x_freq = self.feedforward_freq(x_freq)
        
        return x_freq, attn_freq        


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_freq=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_freq = norm_freq

    def forward(self, x, attn_mask=None, **kwargs):
        attns = []
        x_freq = x  
               
        for i, attn_layer in enumerate(self.attn_layers):
            x, attns_i = attn_layer(x_freq, attn_mask=attn_mask, **kwargs)       
            x_freq = x
            attns.append(attns_i)
            
        if self.norm_freq is not None:
            x_freq = self.norm_freq(x_freq)

        return x_freq, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_freqspar_att, cross_freqspar_att, dsi_layer, d_model,
                 d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()

        self.self_freqspar_att = self_freqspar_att
        self.cross_freqspar_att = cross_freqspar_att
        self.dsi_layer = dsi_layer

        self.feedforward_freq = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout, activation=activation) 
        self.feedforward_trend = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout, activation=activation)
        self.norm_freq1 = nn.LayerNorm(d_model)   
        self.norm_freq2 = nn.LayerNorm(d_model)
        self.norm_trend = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cross_freq, dec_trend, x_mask=None, cross_mask=None, **kwargs):
        x_freq =  x
        
        freq_out, attn_sa_freq = self.self_freqspar_att(x_freq, x_freq, x_freq, x_mask)
        x_freq = self.norm_freq1(x_freq + self.dropout(freq_out))      
        
        freq_out, attn_cr_freq = self.cross_freqspar_att(x_freq, cross_freq, cross_freq, x_mask)   
        x_freq = self.norm_freq2(x_freq + self.dropout(freq_out))     
        
        x_trend = dec_trend
        trend_out = self.dsi_layer(x_trend)         
        
        x_freq = self.feedforward_freq(x_freq)      

        return [x_freq, trend_out], [attn_sa_freq, attn_cr_freq]      


class Decoder(nn.Module):
    def __init__(self, attn_layers, norm_freq=None, norm_trend=None, mlp_out=None, out_projection=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_freq = norm_freq
        self.norm_trend = norm_trend
        self.mlp_out = mlp_out
        self.out_projection = out_projection    

    def forward(self, x, cross_freq, dec_trend, x_mask=None, cross_mask=None, **kwargs):
        attn = []
        x_freq = x
        x_trend = dec_trend
        for i, attn_layer in enumerate(self.attn_layers):
            x, attns_i = attn_layer(x_freq, cross_freq, x_trend,
                                    x_mask=x_mask, cross_mask=cross_mask, **kwargs)
            x_freq, x_trend = x
            attn.append(attns_i)

        return [x_freq, x_trend], attn
    