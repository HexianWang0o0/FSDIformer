import argparse
import torch        
from exp.exp_main import Exp_Main
import random
import numpy as np          


def main():
    parser = argparse.ArgumentParser(description='FSDIformer')

    fix_seed = 2025
    random.seed(fix_seed)   
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)     

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='WPF')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='training or inference')
    parser.add_argument('--model', type=str, required=True, default='FSDIformer',help='model name, options: [FSDIformer]')   
    parser.add_argument('--plot_loss', type=int, default=0, help='whether to plot loss during training.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        
    # data loader
    parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='windfarm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate,'
                             'MS:multivariate predict univariate')  
    parser.add_argument('--target', type=str, default='wind_power') 
    parser.add_argument('--freq', type=str, default='15min',help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')      
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')   
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')

    # Common Parameters in Transformer-based Models
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # FSDIformer specific parameters
    parser.add_argument('--num_decomp', type=int, default=4, help='Number of wavelet decompositions')  
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for the 1DConv value embedding')  
    parser.add_argument('--selected_freq_count', type=int, default=32, help='Number of random selected frequencies')
    parser.add_argument('--hidden_size', type=int, default=1)
    parser.add_argument('--stacks', type=int, default=2)        
    parser.add_argument('--levels', type=int, default=3)       
    parser.add_argument('--concat', type=int, default=0)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7') 

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')

    # Other parameters
    parser.add_argument('--save_flag', type=int, default=0, help='save test result or not')
    parser.add_argument('--save_figure', type=int, default=1, help='save pred-true plot during test or not')
    parser.add_argument('--des', type=str, default='test', help='exp description')

    args = parser.parse_args()
    
    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_dp{}_sfc{}_nl{}_bs{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}_{}'.format(
                args.task_name,
                args.model,
                args.data_path,
                args.selected_freq_count,
                args.num_levels,
                args.batch_size,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.d_layers,
                args.des, ii)

            exp = Exp(args) 
            exp.train(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        for ii in range(args.itr):
            setting = '{}_{}_dp{}_sfc{}_nl{}_bs{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}_{}'.format(
                args.task_name,
                args.model,
                args.data_path,
                args.selected_freq_count,
                args.num_levels,
                args.batch_size,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.d_layers,
                args.des, ii)

        exp = Exp(args)  
        exp.test(setting)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

