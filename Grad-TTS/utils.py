# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import glob
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4
from pymo.writers import BVHWriter


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


def generate_motion_visualization(
    audio, audio_filename, motion, motion_filename, motion_visualizer_pipeline, bvh_filename=None
):
    """

    Args:
        audio (_type_): (1, T_audio)
        audio_filename (_type_): str/path
        motion (_type_): (T_motion, 45)
        motion_filename (_type_): str/path
        motion_visualizer_pipeline (_type_): Pipeline
        bvh_filename (_type_, optional): str/path. Defaults to None.
    """
    audio_filename = Path(audio_filename)
    motion_filename = Path(motion_filename)

    sf.write(audio_filename, audio.flatten(), 22500, "PCM_24")

    # Add motion target
    bvh_values = motion_visualizer_pipeline.inverse_transform([motion])

    if bvh_filename is not None:
        # Write input bvh file
        writer = BVHWriter()
        with open(bvh_filename, "w") as f:
            writer.write(bvh_values[0], f)

    # To stickfigure
    X_pos = MocapParameterizer("position").fit_transform(bvh_values)

    render_mp4(X_pos[0], motion_filename.with_suffix(".temp.mp4"), axis_scale=200)
    combine_video_audio(
        motion_filename.with_suffix(".temp.mp4"),
        audio_filename,
        motion_filename,
    )
    
    

def combine_video_audio(video_filename, audio_filename, final_filename):
    command = f"ffmpeg -i {video_filename} -i {audio_filename} -c:v copy -c:a aac {final_filename} -y"
    subprocess.check_call(command, shell=True)
    Path(video_filename).unlink()
    Path(audio_filename).unlink()
    
def get_dir_without_underscore(variable):
    return [x for x in dir(variable) if x[0] != "_"]
    
def compare_parameters(new, old):
    new_parameters = {}
    modified_parameters = {}
    new_params = get_dir_without_underscore(new)
    old_params = get_dir_without_underscore(old)
    for param in new_params:
        if param in old_params:
            new_param = getattr(new, param)
            old_param = getattr(old, param)
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
            

def module_to_namespace(module):
    return Namespace(**{k: getattr(module, k) for k in get_dir_without_underscore(module) if not k.startswith('_')})


def normalize(data, mu, std):
    return (data - mu) / std

def denormalize(data, mu, std):
    return data * std + mu


def keep_top_k_checkpoints(logdir, checkpoint_dict, epoch, k=5):
    sorted_files = sorted(list(Path(logdir).glob('grad_*.pt')), key=lambda x: int(str(x.stem).split('_')[1]), reverse=True)
    for file in sorted_files[k:]:
        file.unlink()
    torch.save(checkpoint_dict, Path(logdir) / f'grad_{epoch}.pt')
    
    
    
if __name__ == '__main__':
    keep_top_k_checkpoints('logs/test', {'test': 1, 'test2': 2})
    