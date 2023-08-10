import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
from functools import partial
from torch.optim.lr_scheduler import StepLR, OneCycleLR

from nn_module.encoder_module import Encoder1D
from nn_module.decoder_module import PointWiseDecoder1D
from nn_module.hyena_module import hyena1d

from loss_fn import rel_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import datetime
import logging
import shutil
from typing import Union
from einops import rearrange
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, TensorDataset


# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model(res) -> (Encoder1D, PointWiseDecoder1D):
    # currently they are hard coded
    encoder = Encoder1D(
        2,   # u + x coordinates
        128,
        128,
        4,
        res=res
    )

    decoder = PointWiseDecoder1D(
        128,
        1,
        3,
        scale=2,
        res=res
    )
    hyena_bottleneck = hyena1d(128,length = res)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                sum(p.numel() for p in decoder.parameters() if p.requires_grad) + \
                sum(p.numel() for p in hyena_bottleneck.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder,hyena_bottleneck


def make_image_grid(init: torch.Tensor, sequence: torch.Tensor, gt: torch.Tensor, out_path, nrow=8):
    b, n, c = sequence.shape   # c = 1

    init = init.detach().cpu().squeeze(-1).numpy()
    sequence = sequence.detach().cpu().squeeze(-1).numpy()
    gt = gt.detach().cpu().squeeze(-1).numpy()

    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b//nrow, nrow),  # creates 8x8 grid of axes
                     )
    x = np.linspace(0, 1, n)

    for ax, im_no in zip(grid, np.arange(b)):
        # Iterating over the grid returns the Axes.
        # ax.plot(x, init[im_no], c='b', alpha=0.2)
        ax.plot(x, sequence[im_no], c='r')
        ax.plot(x, gt[im_no], '--', c='g', alpha=0.8)
        ax.axis('equal')
        ax.axis('off')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# copied from Galerkin Transformer
def central_diff(x: torch.Tensor, h):
    # assuming PBC
    # x: (batch, seq_len, feats), h is the step size

    pad_0, pad_1 = x[:, -2:-1], x[:, 1:2]
    x = torch.cat([pad_0, x, pad_1], dim=1)
    x_diff = (x[:, 2:] - x[:, :-2])/2  # f(x+h) - f(x-h) / 2h
    # pad = np.zeros(x_diff.shape[0])

    # return np.c_[pad, x_diff/h, pad]
    return x_diff/h


def pad_pbc(x: torch.Tensor, pos: torch.Tensor, h, pad_ratio=1/128):
    # x: (batch, seq_len, feats), h is the step size
    # assuming x in the order of x-axis [0, 1]
    n = x.shape[1]
    pad_0, pad_1 = x[:, -int(pad_ratio*n)-1:-1], x[:, 1:int(pad_ratio*n)+1]
    offset = np.arange(1, int(pad_ratio*n)+1, dtype=np.float32)*h
    pos_pad_0 = 0 - offset[::-1]
    pos_pad_1 = 1 + offset
    pos_pad_0 = rearrange(torch.as_tensor(pos_pad_0).to(pos.device), 'n -> 1 n 1').repeat([x.shape[0], 1, 1])
    pos_pad_1 = rearrange(torch.as_tensor(pos_pad_1).to(pos.device), 'n -> 1 n 1').repeat([x.shape[0], 1, 1])
    return torch.cat([pad_0, x, pad_1], dim=1), torch.cat([pos_pad_0, pos, pos_pad_1], dim=1)


def get_arguments(parser):
    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='Specifies learing rate for optimizer. (default: 1e-3)'
    )
    parser.add_argument(
        '--resume', action='store_true', help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_resume', type=str, default='', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--iters', type=int, default=100000, help='Number of training iterations. (default: 100k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5000, help='Save model checkpoints every x iterations. (default: 5k)'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Size of each batch (default: 16)'
    )
    
    parser.add_argument(
        '--resolution', type=int, default=2048, help='The interval of when sample snapshots from sequence'
    )
    parser.add_argument(
        '--save_name', type=str, default='navier_1e5', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--device', type=str, default='cuda:1', help='Path to log, save checkpoints. '
    )
    return parser


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a PDE transformer")
    parser = get_arguments(parser)
    opt = parser.parse_args()
    print('Using following options')
    print(opt)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()

    # add code for datasets

    print('Preparing the data')
    res = opt.resolution
    dx = 1./res
    ntrain = 9000
    ntest = 1000
    device = opt.device

    batch_size = opt.batch_size
    learning_rate = 0.001
    epochs = 500
    iterations = epochs*(ntrain//batch_size)

    from utils import *
    ################################################################
    # read data
    ################################################################
    flnm=  'ReacDiff_Nu2.0_Rho1.0.hdf5'
    train_data = FNODatasetSingle(flnm,
                            reduced_resolution=4,
                            reduced_resolution_t=1,
                            reduced_batch=1,
                            initial_step=5,
                            saved_folder = "./"
                            )
    val_data = FNODatasetSingle(flnm,
                            reduced_resolution=4,
                            reduced_resolution_t=1,
                            reduced_batch=1,
                            initial_step=5,
                            saved_folder = "./",
                            if_test=True,
                            )
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                num_workers=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                num_workers=8, shuffle=False)
    # instantiate network
    x,y,g = next(iter(train_dataloader))
    print("x:-",x.shape)
    print('Building network')
    encoder, decoder, hyena_bottleneck = build_model(res)
    if use_cuda:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        hyena_bottleneck = hyena_bottleneck.to(device)

    
    # create optimizers
    enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=1e-8)
    dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=1e-8)
    hyena_optim = torch.optim.Adam(list(hyena_bottleneck.parameters()), lr=opt.lr, weight_decay=1e-8)
    enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(enc_optim, T_max=opt.iters)
    dec_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dec_optim, T_max=opt.iters)
    hyena_scheduler=  torch.optim.lr_scheduler.CosineAnnealingLR(hyena_optim, T_max=opt.iters)

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    # if opt.resume:
    #     print(f'Resuming checkpoint from: {opt.path_to_resume}')
    #     ckpt = load_checkpoint(opt.path_to_resume)  # custom method for loading last checkpoint
    #     encoder.load_state_dict(ckpt['encoder'])
    #     decoder.load_state_dict(ckpt['decoder'])

    #     start_n_iter = ckpt['n_iter']

    #     enc_optim.load_state_dict(ckpt['enc_optim'])
    #     dec_optim.load_state_dict(ckpt['dec_optim'])

    #     enc_scheduler.load_state_dict(ckpt['enc_sched'])
    #     dec_scheduler.load_state_dict(ckpt['dec_sched'])
    #     print("last checkpoint restored")

    # now we start the main loop
    n_iter = start_n_iter

    # mixed-precision
    # [encoder, decoder], [enc_optim, dec_optim] = amp.initialize(
    #     [encoder, decoder], [enc_optim, dec_optim], opt_level='O0')


    # for loop going through dataset
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        train_data_iter = iter(train_dataloader)

        while True:

            encoder.train()
            decoder.train()
            hyena_bottleneck.train()
            start_time = time.time()

            try:
                data = next(train_data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                del train_data_iter
                train_data_iter = iter(train_dataloader)
                data = next(train_data_iter)

            # data preparation
            xx,yy,grid = data
            xx = xx[..., 0 , :]
            yy = yy[..., -1, :]
            #yy_1 = yy[...,100,:]
            #print(xx.shape)
            #print(yy1.shape)
            #print(yy_1.shape)
            #rint(yy1==yy_1)
            #xit()
            
            x, y, grid = xx.to(device), yy.to(device), grid.to(device)
            
            x = torch.cat((x, grid), dim=-1)   # concat coordinates as additional feature

            prepare_time = time.time() - start_time
            z = encoder.forward(x, grid)
            z = hyena_bottleneck.forward(z)
            x_out = decoder.forward(z, grid, grid)

            pred_loss = rel_loss(x_out, y, 2)

            #gt_deriv = central_diff(y, dx)
            #pred_deriv = central_diff(x_out, dx)
            #deriv_loss = rel_loss(pred_deriv, gt_deriv, 2)

            loss = pred_loss #+ 1e-3*deriv_loss
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            hyena_optim.zero_grad()
            loss.backward()
            # with amp.scale_loss(loss, [enc_optim, dec_optim]) as scaled_loss:
            #     scaled_loss.backward()
            # print(torch.max(decoder.decoding_transformer.attn_module1.to_q.weight.grad))
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            # Unscales gradients and calls
            enc_optim.step()
            dec_optim.step()
            hyena_optim.step()
            hyena_scheduler.step()
            enc_scheduler.step()
            dec_scheduler.step()
            
            process_time = time.time() - start_time - prepare_time

            pbar.set_description(
                f'Total loss (1e-4): {loss.item()*1e4:.1f}||'
                #f'prediction (1e-4): {pred_loss.item()*1e4:.1f}||'
                #f'derivative (1e-4): {deriv_loss.item()*1e4:.1f}||'
                f'Iters: {n_iter}/{opt.iters}')

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if (n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                print('Testing')

                encoder.eval()
                decoder.eval()
                hyena_bottleneck.eval()
                with torch.no_grad():
                    all_avg_loss = []
                    all_acc_loss = []
                    visualization_cache = {
                        'in_seq': [],
                        'pred': [],
                        'gt': [],
                    }
                    picked = 0
                    for j, data in enumerate(tqdm(test_dataloader)):
                        # data preparation
                        xx,yy,grid = data
                        xx = xx[..., 0 , :]
                        yy = yy[..., -1, :]
            
                        x, y, grid = xx.to(device), yy.to(device), grid.to(device)
                        
                        x = torch.cat((x, grid), dim=-1)
                        # standardize
                        # data_mean = torch.mean(x, dim=1, keepdim=True)
                        # data_std = torch.std(x, dim=1, keepdim=True)
                        # x = (x - data_mean) / data_std
                        # y = (y - data_mean) / data_std
                        data_mean = 0.
                        data_std = 1.

                        prepare_time = time.time() - start_time
                        z = encoder.forward(x, grid)
                        z = hyena_bottleneck.forward(z)
                        x_out = decoder.forward(z, grid,grid)

                        avg_loss = rel_loss(x_out, y, p=2)
                        accumulated_mse = torch.nn.MSELoss(reduction='sum')(x_out*data_std, y*data_std) /   \
                                          (res**2 * x.shape[0])

                        all_avg_loss += [avg_loss.item()]
                        all_acc_loss += [accumulated_mse.item()]

                        # rescale
                        x = x[:, :, :1] * data_std + data_mean
                        x_out = x_out * data_std + data_mean
                        y = y * data_std + data_mean

                        if picked < 64:
                            idx = np.arange(0, min(64 - picked, x.shape[0]))
                            # randomly pick a batch
                            x = x[idx]
                            y = y[idx]
                            x_out = x_out[idx]
                            visualization_cache['gt'].append(y)
                            visualization_cache['in_seq'].append(x)
                            visualization_cache['pred'].append(x_out)
                            picked += x.shape[0]

                all_gt = torch.cat(visualization_cache['gt'], dim=0)
                all_in_seq = torch.cat(visualization_cache['in_seq'], dim=0)
                all_pred = torch.cat(visualization_cache['pred'], dim=0)

                make_image_grid(all_in_seq, all_pred, all_gt,
                                os.path.join("./samples/", f'react_diff_result_iter:{n_iter}_{j}.png'))

                del visualization_cache
                print(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                print(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')

            
                # save checkpoint if needed
                ckpt = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'hyena_bottleneck': hyena_bottleneck.state_dict(),
                    'n_iter': n_iter,
                    'enc_optim': enc_optim.state_dict(),
                    'dec_optim': dec_optim.state_dict(),
                    'hyena_optim': hyena_optim.state_dict(),
                    'enc_sched': enc_scheduler.state_dict(),
                    'dec_sched': dec_scheduler.state_dict(),
                    'hyena_sched': hyena_scheduler.state_dict(),
                }
                torch.save(ckpt,"./hyena_diff_react_{}_{}_{}_{}.pt".format(opt.resolution,n_iter,opt.save_name,flnm))
                del ckpt
                if n_iter >= opt.iters:
                    break
