r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from data import TextMelBatchCollate, TextMelDataset


def get_data_parameters_for_flat_start(train_loader):

    # N related information, useful for transition prob init
    total_mel_len = 0

    # Useful for data mean and data std
    total_mel_sum = 0
    total_mel_sq_sum = 0

    # For motion
    total_motion_sum = 0
    total_motion_sq_sum = 0

    print("We first get the mean:")
    start = time.perf_counter()

    for i, batch in enumerate(tqdm(train_loader)):

        total_mel_len += torch.sum(batch['y_lengths'])

        total_mel_sum += torch.sum(batch['y'])
        total_mel_sq_sum += torch.sum(torch.pow(batch['y'], 2))

        motions = batch['y_motion'][:, :45]
        total_motion_sum += motions.sum([0, 2])
        total_motion_sq_sum += torch.sum(torch.pow(motions, 2), dim=[0, 2])

    mel_mean = total_mel_sum / (total_mel_len * params.n_feats)
    mel_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * params.n_feats)) - torch.pow(mel_mean, 2))

    motion_mean = total_motion_sum / (total_mel_len)
    motion_std = torch.sqrt(
        (total_motion_sq_sum / total_mel_len) - torch.pow(motion_mean, 2)
    )

    print("Total Processing Time:", time.perf_counter() - start)
    
    print("Single loop values")
    print("".join(["-"] * 50))
    print("Mel mean: ", mel_mean)
    print("Mel std: ", mel_std)
    print("Motion mean: ", motion_mean)
    print("Motion std: ", motion_std)

    output = {
        "mel_mean": mel_mean.item(),
        "mel_std": mel_std.item(),
        "motion_mean": motion_mean,
        "motion_std": motion_std,
    }

    return output


def main(args):
    
    train_dataset = TextMelDataset(params.train_filelist_path, params.cmudict_path, params.motion_folder, params.add_blank,
                                   params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                                   params.win_length, params.f_min, params.f_max, params.data_parameters)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=32, shuffle=False)

    output = get_data_parameters_for_flat_start(loader)

    # print({k: v.item() if v.numel() == 1 else v for k, v in output.items()})

    torch.save(
        output,
        args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="data_statistics.pt",
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        required=False,
        help="batch size to fetch data properties",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.force:
        print("File already exists. Use -f to force overwrite")
        sys.exit(1)

    main(args)
