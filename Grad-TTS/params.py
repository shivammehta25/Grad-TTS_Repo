# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

# Note: Do not do highlevel imports here as import os; it will make the module unpickleable
from os.path import exists
from warnings import warn

from torch import load

from model.utils import fix_len_compatibility

# data parameters
train_filelist_path = 'data/filelists/cormac_train.txt'
valid_filelist_path = 'data/filelists/cormac_val.txt'
test_filelist_path = 'resources/filelists/ljspeech/test.txt'
cmudict_path = 'resources/cmu_dictionary'
motion_folder = 'data/cormac/processed_sm0_0_86fps'
add_blank = True
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
n_feats = 80 
# n_motions = fix_len_compatibility(45)
n_motions = 45
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
encoder_type="myencoder" # default and myencoder
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 10 

# decoder parameters
dec_dim = 128 
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/{}'
test_size = 4
n_epochs = 10000
batch_size = 32 
learning_rate = 3e-4
seed = 37
save_every = 1
out_size = fix_len_compatibility(2*22050//256) 
# out_size = None

mu_motion_encoder_params = {
                "hidden_channels": 384,
                "d_head": 64,
                "n_layer": 4,
                "n_head": 1,
                "ff_mult": 4,
                "conv_expansion_factor": 2,
                "dropout": 0.1,
                "dropconv": 0.1,
                "dropatt": 0.1,
                "conv_kernel_size": 21,
                "prior_loss": True 
}
decoder_motion_type = "wavegrad" # [wavegrad, gradtts]
motion_reduction_factor = 1 
if exists("data_parameters_of.pt"):
    data_parameters = load("data_parameters_of.pt")
else:
    data_parameters = None
    warn("data_parameters_of.pt not found. Please run generate_data_statistics.py first")