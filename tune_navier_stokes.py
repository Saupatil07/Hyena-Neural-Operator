import torch
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
from tensorboardX import SummaryWriter

from nn_module.encoder_module import SpatialTemporalEncoder2D
from nn_module.decoder_module import PointWiseDecoder2D
from nn_module.hyena_module import hyena1d

from loss_fn import rel_loss, rel_l2norm_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import logging
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model(opt):

    encoder = SpatialTemporalEncoder2D(
        opt.in_channels,
        opt.encoder_emb_dim,
        opt.out_seq_emb_dim,
        opt.encoder_heads,
        opt.encoder_depth,
    )

    decoder = PointWiseDecoder2D(
        opt.decoder_emb_dim,
        opt.out_channels,
        opt.out_step,
        opt.propagator_depth,
        scale=opt.fourier_frequency,
        dropout=0.0,
    )
    hyena_bottleneck = hyena1d(opt.out_seq_emb_dim)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                sum(p.numel() for p in decoder.parameters() if p.requires_grad) + \
                sum(p.numel() for p in hyena_bottleneck.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder,hyena_bottleneck


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor):
    # assuming PBC
    # x: (batch, seq_len, n), h is the step size, assuming n = h*w
    x = rearrange(x, 'b t (h w) -> b t h w', h=64, w=64)
    h = 1./64.
    x = F.pad(x,
            (1, 1, 1, 1), mode='circular')  # [b t h+2 w+2]
    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2*h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2*h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


def make_image_grid(image: torch.Tensor, out_path, nrow=25):
    b, t, h, w = image.shape
    image = image.detach().cpu().numpy()
    image = image.reshape((b*t, h, w))
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(b*t//nrow, nrow),  # creates 2x2 grid of axes
                    )

    for ax, im_no in zip(grid, np.arange(b*t)):
        # Iterating over the grid returns the Axes.
        ax.imshow(image[im_no])
        ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


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


def get_arguments(parser):
    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Specifies learning rate for tuning. (default: 1e-6)'
    )
    parser.add_argument(
        '--iters', type=int, default=5000, help='Number of training iterations. (default: 100k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=1000, help='Save model checkpoints every x iterations. (default: 5k)'
    )

    # general option
    parser.add_argument(
        '--in_seq_len', type=int, default=10, help='Length of input sequence. (default: 10)'
    )
    # model options for encoder

    parser.add_argument(
        '--in_channels', type=int, default=3, help='Channel of input feature. (default: 3)'
    )
    parser.add_argument(
        '--encoder_emb_dim', type=int, default=128, help='Channel of token embedding in encoder. (default: 128)'
    )
    parser.add_argument(
        '--out_seq_emb_dim', type=int, default=128, help='Channel of output feature map. (default: 128)'
    )
    parser.add_argument(
        '--encoder_depth', type=int, default=2, help='Depth of transformer in encoder. (default: 2)'
    )
    parser.add_argument(
        '--encoder_heads', type=int, default=4, help='Heads of transformer in encoder. (default: 4)'
    )

    # model options for decoder
    parser.add_argument(
        '--out_channels', type=int, default=1, help='Channel of output. (default: 1)'
    )
    parser.add_argument(
        '--decoder_emb_dim', type=int, default=128, help='Channel of token embedding in decoder. (default: 128)'
    )
    parser.add_argument(
        '--out_step', type=int, default=10, help='How many steps to propagate forward each call. (default: 10)'
    )
    parser.add_argument(
        '--out_seq_len', type=int, default=40, help='Length of output sequence. (default: 40)'
    )
    parser.add_argument(
        '--propagator_depth', type=int, default=2, help='Depth of mlp in propagator. (default: 2)'
    )
    parser.add_argument(
        '--decoding_depth', type=int, default=2, help='Depth of decoding network in the decoder. (default: 2)'
    )
    parser.add_argument(
        '--fourier_frequency', type=int, default=8, help='Fourier feature frequency. (default: 8)'
    )
    parser.add_argument(
        '--use_grad', action='store_true',
    )
    parser.add_argument(
        '--curriculum_steps', type=int, default=0, help='at initial stage, dont rollout too long'
    )
    parser.add_argument(
        '--curriculum_ratio', type=float, default=0.2, help='how long is the initial stage?'
    )
    parser.add_argument(
        '--aug_ratio', type=float, default=0.0, help='Probability to randomly crop'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Size of each batch (default: 16)'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='Path to dataset.'
    )

    parser.add_argument(
        '--train_seq_num', type=int, default=50, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_seq_num', type=int, default=100, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--save_name', type=str, default='navier_1e5', help='Path to log, save checkpoints. '
    )
    return parser


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a PDE transformer")
    parser = get_arguments(parser)
    opt = parser.parse_args()
    print('Using following options')
    print(opt)
    # add code for datasets
    device = opt.device
    print('Preparing the data')

    # instantiate network
    print('Building network')
    encoder, decoder, hyena_bottleneck = build_model(opt)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        hyena_bottleneck = hyena_bottleneck.to(device)

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    checkpoint_dir = os.path.join(opt.log_dir, 'model_ckpt')
    #ensure_dir(checkpoint_dir)

    sample_dir = os.path.join(opt.log_dir, 'samples')
    #ensure_dir(sample_dir)

    # save option information to the disk
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (opt.log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')
    for arg in vars(opt):
        logger.info(f'{arg}: {getattr(opt, arg)}')

    # load checkpoint if needed/ wanted
    start_n_iter = 0

    # create optimizers

    enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=1e-8)
    dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=1e-8)
    hyena_optim = torch.optim.Adam(list(hyena_bottleneck.parameters()), lr=opt.lr, weight_decay=1e-8)
    enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(enc_optim, T_max=opt.iters)
    dec_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dec_optim, T_max=opt.iters)
    hyena_scheduler=  torch.optim.lr_scheduler.CosineAnnealingLR(hyena_optim, T_max=opt.iters)
    
    # now we start the main loop
    n_iter = start_n_iter

    data_path = opt.dataset_path
    ntrain = opt.train_seq_num
    ntest = opt.test_seq_num

    data = np.load(data_path)
    print(data.shape)
    x_train = data[:opt.in_seq_len, ..., :ntrain]  # input: a(x)
    y_train = data[opt.in_seq_len:opt.in_seq_len+opt.out_seq_len, ..., :ntrain]  # solution: u(x)

    x_test = data[:opt.in_seq_len, ..., -ntest:]  # input: a(x)
    y_test = data[opt.in_seq_len:opt.in_seq_len+opt.out_seq_len, ..., -ntest:]  # solution: u(x)

    x_train = rearrange(torch.as_tensor(x_train, dtype=torch.float32), 't h w n -> n t (h w)')
    x_test = rearrange(torch.as_tensor(x_test, dtype=torch.float32), 't h w n -> n t (h w)')
    y_train = rearrange(torch.as_tensor(y_train, dtype=torch.float32), 't h w n -> n t (h w)')
    y_test = rearrange(torch.as_tensor(y_test, dtype=torch.float32), 't h w n -> n t (h w)')
    del data

    # gaussian normalization
    x_mean = torch.mean(x_train).unsqueeze(0)   # [1, t_in, hw]
    x_std = torch.std(x_train).unsqueeze(0)     # [1, t_in, hw]

    y_mean = torch.mean(y_train).unsqueeze(0)  # [1, t_out, hw]
    y_std = torch.std(y_train).unsqueeze(0)  # [1, t_out, hw]

    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std

    x_test = (x_test - x_mean) / x_std

    if use_cuda:
        x_mean, x_std, y_mean, y_std = x_mean.to(device), x_std.to(device), y_mean.to(device), y_std.to(device)

    x0, y0 = np.meshgrid(np.linspace(0, 1, 64),
                        np.linspace(0, 1, 64))
    xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
    grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').unsqueeze(0).float()  # [64*64, 2]

    train_dataloader = DataLoader(TensorDataset(x_train, y_train),
                                batch_size=opt.batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test),
                                batch_size=opt.batch_size,
                                shuffle=False)
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
            in_seq, gt = data
            input_pos = prop_pos = repeat(grid, '() n c -> b n c', b=in_seq.shape[0])

            if use_cuda:
                in_seq = in_seq.to(device)
                gt = gt.to(device)

                input_pos = input_pos.to(device)
                prop_pos = prop_pos.to(device)

            in_seq = rearrange(in_seq, 'b t n -> b n t')

            if np.random.uniform() > (1-opt.aug_ratio):
                sampling_ratio = np.random.uniform(0.45, 0.95)
                input_idx = torch.as_tensor(
                    np.concatenate(
                        [np.random.choice(input_pos.shape[1], int(sampling_ratio*input_pos.shape[1]), replace=False).reshape(1,-1)
                        for _ in range(in_seq.shape[0])], axis=0)
                    ).view(in_seq.shape[0], -1).to(device)

                in_seq = index_points(in_seq, input_idx)
                input_pos = index_points(input_pos, input_idx)

            in_seq = torch.cat((in_seq, input_pos), dim=-1)

            z = encoder.forward(in_seq, input_pos)
            z = hyena_bottleneck.forward(z)
            if opt.curriculum_steps > 0 and n_iter < int(opt.curriculum_ratio * opt.iters):
                progress = (n_iter*2) / (opt.iters*opt.curriculum_ratio)
                curriculum_steps = opt.curriculum_steps +\
                                int(max(0,  progress - 1.)*((opt.out_seq_len - opt.curriculum_steps)/2.)) * 2
                gt = gt[:, :curriculum_steps, :]   # [b t n]
                x_out = decoder.rollout(z, prop_pos, curriculum_steps, input_pos)
            else:
                x_out = decoder.rollout(z, prop_pos, opt.out_seq_len, input_pos)

            pred_loss = rel_l2norm_loss(x_out, gt)
        
            loss = pred_loss
            if opt.use_grad:
                gt_grad_x, gt_grad_y = central_diff(gt)
                pred_grad_x, pred_grad_y = central_diff(x_out)
                grad_loss = rel_l2norm_loss(pred_grad_x, gt_grad_x) + \
                            rel_l2norm_loss(pred_grad_y, gt_grad_y)
                loss += 5e-2 * grad_loss
            else:
                grad_loss = torch.tensor([-1.])  # placeholder

            enc_optim.zero_grad()
            dec_optim.zero_grad()
            hyena_optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.)
            torch.nn.utils.clip_grad_norm_(hyena_bottleneck.parameters(), 2.)

            # Unscales gradients and calls
            enc_optim.step()
            dec_optim.step()
            hyena_optim.step()
            hyena_scheduler.step()
            enc_scheduler.step()
            dec_scheduler.step()

            # udpate tensorboardX
            writer.add_scalar('train_loss', loss, n_iter)
            writer.add_scalar('prediction_loss', pred_loss, n_iter)

            pbar.set_description(
                f'Total (1e-4): {loss.item()*1e4:.1f}||'
                f'pred (1e-4): {pred_loss.item()*1e4:.1f}||'
                f'grad (1e-4): {grad_loss.item()*1e4:.1f}||'
                f'lr (1e-3): {enc_scheduler.get_last_lr()[0]*1e3:.4f}||'
                f'Seq len: {gt.shape[1]}||'
                f'Iters: {n_iter}/{opt.iters}'
            )

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if ((n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters):
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()
                hyena_bottleneck.eval()
                with torch.no_grad():
                    all_avg_loss = []
                    all_acc_loss = []
                    all_last_loss = []
                    visualization_cache = {
                        'in_seq': [],
                        'pred': [],
                        'gt': [],
                    }
                    picked = 0
                    for j, data in enumerate(tqdm(test_dataloader)):
                        # data preparation
                        in_seq, gt = data

                        input_pos = prop_pos = repeat(grid, '() n c -> b n c', b=in_seq.shape[0])

                        if use_cuda:
                            in_seq = in_seq.to(device)
                            gt = gt.to(device)

                            input_pos = input_pos.to(device)
                            prop_pos = prop_pos.to(device)

                        in_seq = rearrange(in_seq, 'b t n -> b n t')
                        in_seq = torch.cat((in_seq, input_pos), dim=-1)

                        z = encoder.forward(in_seq, input_pos)
                        z = hyena_bottleneck.forward(z)
                        x_out = decoder.rollout(z, prop_pos, opt.out_seq_len, input_pos)  # [b, seq_len, n]

                        x_out = x_out*y_std + y_mean   # denormalize

                        avg_loss = rel_loss(x_out, gt, p=2)
                        accumulated_mse = torch.nn.MSELoss(reduction='sum')(x_out, gt)/\
                                        (gt.shape[-1] * gt.shape[0])

                        loss_at_last_step = rel_loss(x_out[:, -1:, ...], gt[:, -1:, ...], p=2)

                        all_avg_loss += [avg_loss.item()]
                        all_acc_loss += [accumulated_mse.item()]
                        all_last_loss += [loss_at_last_step.item()]

                        # rescale
                        in_seq = in_seq[:, ..., :-2]

                        # for plotting, we reconstruct them back to the shape of grid
                        in_seq = rearrange(
                            rearrange(in_seq, 'b n t -> b t n') * x_std + x_mean, 'b t (h w) -> b t h w', h=64, w=64)
                        x_out = rearrange(x_out, 'b t (h w) -> b t h w', h=64, w=64)
                        gt = rearrange(gt, 'b t (h w) -> b t h w', h=64, w=64)

                        if picked < 20:
                            idx = np.arange(0, min(20 - picked, in_seq.shape[0]))
                            # randomly pick a batch
                            in_seq = in_seq[idx, ::2]  # chop off the position channels
                            gt = gt[idx, ::2]
                            x_out = x_out[idx, ::2]
                            visualization_cache['gt'].append(gt)
                            visualization_cache['in_seq'].append(in_seq)
                            visualization_cache['pred'].append(x_out)
                            picked += in_seq.shape[0]

                all_gt = torch.cat(visualization_cache['gt'], dim=0)
                all_in_seq = torch.cat(visualization_cache['in_seq'], dim=0)
                all_pred = torch.cat(visualization_cache['pred'], dim=0)

                gt = torch.cat((all_in_seq, all_gt), dim=1) # concatenate in the temporal dimension
                pred = torch.cat((all_in_seq, all_pred), dim=1) # concatenate in the temporal dimension

                #make_image_grid(gt,
                #                os.path.join(sample_dir, f'gt_iter:{opt.save_name}_{n_iter}_{j}.png'), nrow=gt.shape[1])

                #make_image_grid(pred,
                #                os.path.join(sample_dir, f'pred_iter:{opt.save_name}_{n_iter}_{j}.png'), nrow=gt.shape[1])

                del visualization_cache
                writer.add_scalar('testing avg loss', np.mean(all_avg_loss), global_step=n_iter)

                print(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                print(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')
                print(f'Testing loss at the last step (1e-4): {np.mean(all_last_loss)*1e4}')

                logger.info(f'Current iteration: {n_iter}')
                logger.info(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                logger.info(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')
                logger.info(f'Testing loss at the last step (1e-4): {np.mean(all_last_loss)*1e4}')

            
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

                torch.save(ckpt,"./navier_hyena_{}_{}.pt".format(opt.save_name,n_iter))
                del ckpt
                if n_iter >= opt.iters:
                    print('Running finished...')
                    exit()
