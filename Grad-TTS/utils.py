# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import glob
import os
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

def get_dir_without_underscore(variable):
    return [x for x in dir(variable) if x[0] != "_"]

def recursively_change_tensor_to_list(variable):
    if isinstance(variable, torch.Tensor):
        return variable.tolist()
    elif isinstance(variable, dict):
        return {key: recursively_change_tensor_to_list(value) for key, value in variable.items()}
    elif isinstance(variable, list):
        return [recursively_change_tensor_to_list(value) for value in variable]
    else:
        return variable
    

def compare_parameters(new, old):
    new_parameters = {}
    modified_parameters = {}
    new_params = get_dir_without_underscore(new)
    old_params = get_dir_without_underscore(old)
    for param in new_params:
        if param in old_params:
            new_param = recursively_change_tensor_to_list(getattr(new, param))
            old_param = recursively_change_tensor_to_list(getattr(old, param))

            if new_param != old_param:
                modified_parameters[param] = {
                    'new': new_param,
                    'old': old_param
                    }
            old_params.remove(param)
        else:
            new_parameters[param] = new_param
    
    if len(modified_parameters) > 0: 
        print('Modified parameters:')
        for key, param in modified_parameters.items():
            print(f"\t{key}:")
            print(f"\t\tOld: {param['old']}")
            print(f"\t\tNew: {param['new']}")
    
    if len(new_parameters) > 0:
        print('New parameters:')
        for key, param in new_parameters.items():
            print(f"\t{key}:")
            print(f"\t\t{param}")
    
    if len(old_params) > 0:
        print('Removed parameters:')
        for param in old_params:
            print(f"\t{param} : {getattr(old, param)}")
            

def module_to_namespace(module, modules_to_ignore=('torch', 'torch.nn', 'torch.nn.functional', 'warnings')):
    return Namespace(**{k: getattr(module, k) for k in get_dir_without_underscore(module) if not k.startswith('_') and k not in modules_to_ignore})


def keep_top_k_checkpoints(logdir, checkpoint_dict, epoch, k=5):
    if k is not None:
        sorted_files = sorted(list(Path(logdir).glob('grad_*.pt')), key=lambda x: int(str(x.stem).split('_')[1]), reverse=True)
        for file in sorted_files[k - 1:]:
            file.unlink()
    torch.save(checkpoint_dict, Path(logdir) / f'grad_{epoch}.pt')
    
    
def save_every_n(logdir, checkpoint_dict, epoch, n=100):
    if epoch % n == 0:
        torch.save(checkpoint_dict, Path(logdir) / f'epoch_{epoch}.pt')
    