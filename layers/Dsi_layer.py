import math
import torch
import numpy as np
from torch import nn
from layers.Functionality import MLPLayer


# This code is adapted from SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction.
# Paper link: https://doi.org/10.48550/arXiv.2106.09305
# Original license: MIT (or other)
# Modifications and improvements are made to better fit wind power data and improve performance on wind power 
# forecasting.


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def first_half(self, x):
        seq_len = x.size(1)
        return x[:, :seq_len // 2, :]  # first half

    def second_half(self, x):
        seq_len = x.size(1)
        return x[:, seq_len // 2:, :]  # second half

    def forward(self, x):
        '''Returns the first and second halves along the sequence length'''
        return (self.first_half(x), self.second_half(x))


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel = 5, dropout=0.5, groups = 1, hidden_size = 1):
        super(Interactor, self).__init__() 
        self.kernel_size = kernel 
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups            
        if self.kernel_size % 2 == 0:        
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 # by default: stride == 1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1     # by default: stride == 1    

        else:       
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1     
        self.splitting = splitting      
        self.split = Splitting()                    

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [  
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),   
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout),  # Spatial Dropout for 
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),        
            nn.ReLU()  
        ]       
        modules_U += [  
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),   
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout),  # Spatial Dropout
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),        
            nn.ReLU()  
        ]  

        modules_phi += [  
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),   
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout),  # Spatial Dropout 
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),        
            nn.ReLU()  
        ]            
        modules_psi += [        
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout),  # Spatial Dropout
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.ReLU()   
        ]                          
        self.phi = nn.Sequential(*modules_phi)              
        self.psi = nn.Sequential(*modules_psi)    
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)          

    def forward(self, x):

        (x_first_half, x_second_half) = self.split(x) 
        
        x_first_half = x_first_half.permute(0, 2, 1)           # (batch_size, dim, seq_len)        
        x_second_half = x_second_half.permute(0, 2, 1)         # (batch_size, dim, seq_len) 

        d = x_second_half.mul(torch.exp(self.phi(x_first_half)))
        c = x_first_half.mul(torch.exp(self.psi(x_second_half)))

        x_first_half = c + self.U(d)
        x_second_half = d - self.P(c)

        # add sigmoid activation
        x_first_half = torch.sigmoid(x_first_half)
        x_second_half = torch.sigmoid(x_second_half)

        return (x_first_half, x_second_half)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes=in_planes, splitting=True,
                 kernel=kernel, dropout=dropout, groups=groups, hidden_size=hidden_size)

    def forward(self, x):
        (x_first_update, x_second_update) = self.level(x)
        return (x_first_update.permute(0, 2, 1), x_second_update.permute(0, 2, 1)) # (batch_size, dim, seq_len) --> (batch_size, seq_len, dim)  

    
class EncoderTree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size):
        super().__init__()

        self.current_level = current_level
        
        self.workingblock = InteractorLevel(
            in_planes = in_planes,
            kernel = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            )  

        if current_level!=0:
            self.Tree_first_half=EncoderTree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size)
            self.Tree_second_half=EncoderTree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size)


    def zip_up_the_pants(self, first_half, second_half):
        return torch.cat([first_half, second_half], dim=1)   # B, L, D    


    def forward(self, x):
        x_first_update, x_second_update = self.workingblock(x)
        
        if self.current_level == 0: 
            return self.zip_up_the_pants(x_first_update, x_second_update)            
        else:   
            return self.zip_up_the_pants(self.Tree_first_half(x_first_update), self.Tree_second_half(x_second_update))

        
class Dsilayer(nn.Module):
    def __init__(self, input_len=64, input_dim=3, hid_size=128, num_stacks=2,
                 num_levels=3, concat_len=0, groups=1, kernel=5, dropout=0.5, out_len=4):
        super(Dsilayer, self).__init__()
        
        self.input_len = input_len 
        self.input_dim = input_dim
        self.hidden_size = hid_size
        self.num_levels = num_levels  
        self.groups = groups                  
        self.kernel_size = kernel        
        self.dropout = dropout           
        self.concat_len = concat_len       
        self.stacks = num_stacks                             
        self.output_len = out_len           
        
        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            current_level=self.num_levels - 1,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,   
            hidden_size=self.hidden_size,
            )  

        if self.stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
                current_level=self.num_levels - 1,
                kernel_size=self.kernel_size,
                dropout=self.dropout,  
                groups=self.groups,       
                hidden_size=self.hidden_size,    
                )        

        for m in self.modules():   
            if isinstance(m, nn.Conv2d):  
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels                           
                m.weight.data.normal_(0, math.sqrt(2. / n))             
            elif isinstance(m, nn.BatchNorm2d):         
                m.weight.data.fill_(1)      
                m.bias.data.zero_()     
            elif isinstance(m, nn.Linear):               
                m.bias.data.zero_()       

        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)                           

        if self.stacks == 2:
            if self.concat_len:
                self.projection2 = nn.Conv1d(self.concat_len + self.output_len,self.output_len,
                                            kernel_size = 1, bias = False)      
            else:       
                self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)                     

        self.feedforward_trend = MLPLayer(d_model=self.input_dim, d_ff=512*4, kernel_size=1, dropout=0.05, activation='gelu')
        self.norm_trend = nn.LayerNorm(self.input_dim)      


    def forward(self, x, *_, **__):
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)     

        # the first stack       
        res1 = x            # (batch_size, seq_len, dim)     
        x = self.blocks1(x)     
        x += res1                                                                  
        x = self.projection1(x)

        if self.stacks == 1:     
            return x

        elif self.stacks == 2:      
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)  

            # the second stack
            res2 = x
            x = self.blocks2(x)  
            x += res2   
            x = self.norm_trend(x)   
            
            x = self.feedforward_trend(x)
            x = self.projection2(x) 
            
            return x       
