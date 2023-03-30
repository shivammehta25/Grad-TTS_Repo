# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import params
from data import TextMelBatchCollate, TextMelDataset
from model import GradTTS
from text.symbols import symbols
from utils import (compare_parameters, module_to_namespace, plot_tensor,
                   save_plot)

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
motion_folder = params.motion_folder

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_motions = params.n_motions
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

mu_motion_encoder_params = params.mu_motion_encoder_params
decoder_motion_type = params.decoder_motion_type
motion_reduction_factor = params.motion_reduction_factor
data_parameters = params.data_parameters

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train GradTTS')
    parser.add_argument('--only-speech', '-s', action='store_true', help='Train without motion')
    parser.add_argument('--resume_from_checkpoint', '-c', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('-run_name', '-r', type=str, required=True, help='Name of the run')
    args = parser.parse_args()      
    
    log_dir = log_dir.format(args.run_name)
    print(f'Running : {args.run_name}')
    
    if args.only_speech:
        print('Note*: Only speech flag is True. training only speech model')
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelDataset(train_filelist_path, cmudict_path, motion_folder, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max, data_parameters)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, motion_folder, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max, data_parameters)

    print('Initializing model...')
    model = GradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, n_motions, dec_dim, beta_min, beta_max, pe_scale, 
                    mu_motion_encoder_params, decoder_motion_type, motion_reduction_factor,args.only_speech).cuda()
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate) 
    
    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y']
        motion = item['y_motion']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')
        if not args.only_speech: 
            logger.add_image(f'image_{i}/ground_truth_motion', plot_tensor(motion.squeeze()),
                            global_step=0, dataformats='HWC')
            save_plot(motion.squeeze(), f'{log_dir}/original_motion_{i}.png')
            
    if args.resume_from_checkpoint is not None:
        print('[*] Loading checkpoint from {}'.format(args.resume_from_checkpoint))
        ckpt = torch.load(args.resume_from_checkpoint)    
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        compare_parameters(params, ckpt['params'])
        iteration = ckpt['iteration']
        start_epoch = ckpt['epoch']
    else:
        iteration = 0
        start_epoch = 1

    print('Start training...')
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                y_motion = batch['y_motion'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     y_motion, 
                                                                     out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                
                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % params.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, y_motion_enc, y_motion_dec, attn = model(x, x_lengths, n_timesteps=20)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                if not args.only_speech:
                    logger.add_image(f'image_{i}/generated_enc',
                                    plot_tensor(y_motion_enc.squeeze().cpu()),
                                    global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/generated_dec',
                                    plot_tensor(y_motion_dec.squeeze().cpu()),
                                    global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec_{i}.png')
                if not args.only_speech:
                    save_plot(y_motion_enc.squeeze().cpu(), 
                            f'{log_dir}/generated_enc_motion_{i}.png')
                    save_plot(y_motion_dec.squeeze().cpu(), 
                            f'{log_dir}/generated_dec_motion_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_dir}/alignment_{i}.png')

        ckpt = {
            'model': model.state_dict(),
            'params': module_to_namespace(params),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'iteration': iteration,
        }
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
