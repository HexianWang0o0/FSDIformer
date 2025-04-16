from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic     
from models import FSDIformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, PlotLossesSame
from utils.metrics import metric  

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

from torch.optim import lr_scheduler

import pickle
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import os
import time

import warnings
import numpy as np


warnings.filterwarnings('ignore')       


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)


    def _build_model(self):
        model_dict = {
            'FSDIformer': FSDIformer, 
        }   

        model = model_dict[self.args.model].Model(self.args).float()        
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)     
        return data_set, data_loader

    
    def vali(self, args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
        total_loss = 0.0
        total_mae_loss = 0.0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
                
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

                f_dim = -1 
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                loss = criterion(outputs, batch_y)
                mae_loss = mae_metric(outputs, batch_y)

                gathered_loss = accelerator.gather(loss)
                gathered_mae_loss = accelerator.gather(mae_loss)

                total_loss += gathered_loss.sum().item()
                total_mae_loss += gathered_mae_loss.sum().item()
                total_samples += gathered_loss.numel()

            avg_mse_loss = total_loss / total_samples
            avg_mae_loss = total_mae_loss / total_samples

        model.train()
        return avg_mse_loss, avg_mae_loss


    def train(self, setting):
               
        hf_ds_config_ = './ds_config_zero1.json'

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config_)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

        accelerator.print('>>>>>>>>>>training begins<<<<<<<<<<<<')
        accelerator.print(self.args)
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        train_batch = next(iter(train_loader))
        valid_batch = next(iter(vali_loader))
        accelerator.print(f"Train dataset size: {len(train_data)}")
        accelerator.print(f"Valid dataset size: {len(vali_data)}")
        accelerator.print(f"Train Loader size: {len(train_loader)}")
        accelerator.print(f"Valid Loader size: {len(vali_loader)}")
        accelerator.print(f"Train batch shape: inputs {train_batch[0].shape}, outputs {train_batch[1].shape}")
        accelerator.print(f"Valid batch shape: inputs {valid_batch[0].shape}, outputs {valid_batch[1].shape}")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)   

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=self.args.patience, verbose=True)

        trained_parameters = []
        for p in self.model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p) 

        model_optim = optim.Adam(trained_parameters, lr=self.args.learning_rate)

        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        elif self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        else:
            scheduler = None

        train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, self.model, model_optim, scheduler)
        train_steps = len(train_loader)
        
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            attens = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):   
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
                
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                       
                f_dim = -1 
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())
                if (i + 1) % 50 == 0:
                    elapsed_time = time.time() - epoch_time
                    speed = (i + 1) / elapsed_time
                    accelerator.print("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f} | speed: {4:.2f} iters/sec".format(i + 1, train_steps, epoch + 1, np.average(train_loss), speed))
                
                accelerator.backward(loss)
                model_optim.step()  

                if self.args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  
            accelerator.print("Validing...")    
            vali_loss, vali_mae_loss = self.vali(self.args, accelerator, model, vali_data, vali_loader, criterion,mae_metric)       
            accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Vali MAE: {3:.7f}".format(epoch + 1, train_loss,vali_loss, vali_mae_loss))

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if self.args.lradj != 'TST':
                if self.args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:       
                    if epoch == 0:
                        self.args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))                

            # Plot the losses:
            if self.args.plot_loss and accelerator.is_main_process:
                loss_save_dir = path + '/pic/train_loss.png'
                loss_save_dir_pkl = path + '/train_loss.pickle'
                if os.path.exists(loss_save_dir_pkl):
                    fig_progress = pickle.load(open(loss_save_dir_pkl, 'rb'))

                if 'fig_progress' not in locals():
                    fig_progress = PlotLossesSame(epoch + 1,
                                                  Training=train_loss,
                                                  Validation=vali_loss) 
                else:
                    fig_progress.on_epoch_end(Training=train_loss,
                                              Validation=vali_loss)    

                if not os.path.exists(os.path.dirname(loss_save_dir)):
                    os.makedirs(os.path.dirname(loss_save_dir))
                fig_progress.fig.savefig(loss_save_dir)    
                pickle.dump(fig_progress, open(loss_save_dir_pkl, 'wb'))    # To load figure that we can append to
        accelerator.wait_for_everyone()
        accelerator.free_memory()
        accelerator.print('================Training has Finished!===============')
        return True


    def test(self, setting):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'testing is deployed on {device}')
       
        test_data, test_loader = self._get_data(flag='test')

        test_batch = next(iter(test_loader))

        print(f"Test dataset size: {len(test_data)}")
        print(f"Test loader size: {len(test_loader)}")
        print(f"Test batch shape: inputs {test_batch[0].shape}, outputs {test_batch[1].shape}")

        load_model_path = os.path.join(self.args.checkpoints, setting)  # unique checkpoint saving path
        load_model_path = load_model_path + '/' + 'checkpoint.pth'

        torch.cuda.empty_cache()
        print('Loading model...')
        if os.path.exists(load_model_path):
            self.model.load_state_dict(torch.load(load_model_path))
            print('Load model successful!')
        else:
            print('Model path does not exist!')
            return

        all_predictions = []
        all_targets = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model = self.model.to(device)

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(device)
                
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                targets = batch_y[:, -self.args.pred_len:, f_dim:]

                all_predictions.append(outputs.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

                if i % 50 == 0:
                    batch_x_np = batch_x.detach().cpu().numpy()
                    true_np = targets.detach().cpu().numpy()
                    pred_np = outputs.detach().cpu().numpy()
                    gt = np.concatenate((batch_x_np[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((batch_x_np[0, :, -1], pred_np[0, :, -1]), axis=0)
                    if self.args.save_figure:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        print('Prediction shape:', all_predictions.shape, 'Truth shape:', all_targets.shape)
        mae, mse = metric(all_predictions, all_targets)
        print('mse:{}, mae:{}'.format(mse, mae))
        losses = {
            'mse': mse,
            'mae': mae,
        }
        with open(folder_path + "results_loss.txt", 'w') as f:
            for key, value in losses.items():
                f.write('%s:%s\n' % (key, value))
        if self.args.save_flag:
            np.savez("test_results.npz", predictions=all_predictions, targets=all_targets)

        print('================Test finished==================')

