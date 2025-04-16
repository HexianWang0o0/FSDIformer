import torch
import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter


def apply_hpfilter(data, lamb=10):
    device = data.device
    batch_size, seq_len, enc_in = data.shape

    I = torch.eye(seq_len, device=device)
    D = torch.zeros(seq_len - 2, seq_len, device=device)
    for i in range(seq_len - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    
    penalty = lamb * (D.T @ D)
    
    Hp_matrix = I + penalty
    
    Hp_matrix_inv = torch.linalg.inv(Hp_matrix) 
    
    trend = torch.einsum('ij,bjk->bik', Hp_matrix_inv, data)  
    
    residual = data - trend

    return trend, residual
