import torch        
import torch.nn as nn
from layers.FSDIformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder
from layers.FreqSparAtten import FreqSparAtten, FreqSparAttn_Layer
from layers.Embed import DataEmbedding  
from layers.WaveletTransform import get_wt   
from layers.Functionality import MLPLayer  
from layers.Dsi_layer import Dsilayer       


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention
        self.num_decomp = configs.num_decomp

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # EncDec Periodic Component Embeddings:
        self.enc_embeddingF = DataEmbedding(self.num_decomp * configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, kernel_size=configs.kernel_size, temp_embed=False, pos_embed=False)      
        self.dec_embeddingF = DataEmbedding(self.num_decomp * configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout, kernel_size=configs.kernel_size, temp_embed=False, pos_embed=False)          

        # Decoder Trend Component Embedding: 
        self.dec_embeddingT = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size, temp_embed=False, pos_embed=False)       

        encoder_self_att = FreqSparAtten(
                                         seq_len_q=self.seq_len,  
                                         seq_len_kv=self.seq_len,
                                         selected_freq_count=configs.selected_freq_count,
                                         )         

        decoder_self_att = FreqSparAtten(
                                         seq_len_q=self.seq_len//2+self.pred_len,  
                                         seq_len_kv=self.seq_len//2+self.pred_len,  
                                         selected_freq_count=configs.selected_freq_count,
                                         )  

        decoder_cross_att = FreqSparAtten(
                                          seq_len_q=self.seq_len//2+self.pred_len,  
                                          seq_len_kv=self.seq_len,
                                          selected_freq_count=configs.selected_freq_count,
                                          )

        self.out_projection=nn.Linear(configs.d_model, 1)
        
        # FSDIformer encoder
        self.encoder = Encoder(    
            [                      
                EncoderLayer(           
                    FreqSparAttn_Layer(       
                        encoder_self_att,
                        configs.d_model, configs.n_heads
                    ),                   
                    configs.d_model,        
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,  
                ) for l in range(configs.e_layers)      
            ],       
            norm_freq=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
        )             
        
        # FSDIformer decoder       
        self.decoder = Decoder(     
            [       
                DecoderLayer(
                    FreqSparAttn_Layer(       
                        decoder_self_att,
                        configs.d_model, configs.n_heads
                    ),
                    FreqSparAttn_Layer(       
                        decoder_cross_att,
                        configs.d_model, configs.n_heads
                    ),            
                    Dsilayer( 
                        input_len=configs.seq_len, out_len=configs.pred_len, input_dim=configs.d_model,hid_size=configs.hidden_size, num_stacks=configs.stacks, num_levels = configs.levels, groups=configs.groups, kernel=configs.kernel, dropout=configs.dropout, original_dim=configs.enc_in,concat_len=configs.concat
                    ),        
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,      
                ) for l in range(configs.d_layers)                
            ],  
            norm_freq=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            mlp_out=MLPLayer(d_model=configs.d_model, d_ff=configs.d_ff, kernel_size=1,
                             dropout=configs.dropout, activation=configs.activation) if configs.mlp_out else None,
            out_projection=nn.Linear(configs.d_model, configs.c_out),  
        )       

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, cross_mask=None, **_):
        # Wavelet Decomposition
        x_enc_freq, x_enc_trend = get_wt(x_enc, num_decomp=self.num_decomp)    # freq:(batch_size, seq_len, enc_in*num_decomp)  e.g.(32, 64, 3*4)  
        x_dec_freq, _ = get_wt(x_dec[:, :-self.pred_len, :], num_decomp=self.num_decomp)

        x_dec_freq_input = x_dec_freq
        dec_freq_place = torch.zeros([x_dec_freq_input.shape[0], self.pred_len, x_dec_freq_input.shape[2]], device=x_dec_freq_input.device)
        x_dec_freq_input = torch.cat([x_dec_freq_input, dec_freq_place], dim=1)
        
        x_enc_freq = self.enc_embeddingF(x_enc_freq, x_mark_enc)    # (batch_size, seq_len, d_model) e.g.(32, 64, 512)  
        x_dec_freq_input = self.dec_embeddingF(x_dec_freq_input, x_mark_dec)    # (batch_size, label_len+pred_len, d_model) e.g.(32, 32+16, 512)
        x_dec_trend = self.dec_embeddingT(x_enc_trend, x_mark_enc)   # (batch_size, seq_len, d_model) e.g.(32, 64, 512)

        attns = []

        enc_freq, a = self.encoder(x_enc_freq, attn_mask=enc_self_mask)  # (batch_size, seq_len, d_model)
        attns.append(a)

        dec_out, a = self.decoder(x_dec_freq_input, enc_freq, x_dec_trend, x_mask=dec_self_mask, cross_mask=cross_mask)
        attns.append(a)  
        
        dec_out_freq, dec_out_trend = dec_out
        dec_out_freq = dec_out_freq[:, -self.pred_len:, :]
        
        output = dec_out_freq + dec_out_trend   # (batch_size, pred_len, d_model)
        output = self.out_projection(output)    # (batch_size, pred_len, 1)

        if self.output_attention:
            return output, attns
        else:   
            return output        
