import torch
import numpy as np
from tqdm import tqdm

import os
from pathlib import Path
import pickle as pickle

from trackintel.geogr.distances import calculate_distance_matrix


def generate_samples(model, config):
    samples = []
    for _ in range(int(config.num_gen_samples / config.batch_size)):
        sample = model.sample(config.batch_size, config.generate_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)


def construct_discriminator_pretrain_dataset(config, train_data, train_idx, all_locs):
    save_path = os.path.join(config.temp_save_root, "temp", "discriminator_fake_dataset.pk")
    # if the file is pre-generated we load the file, otherwise run self.generate_data()
    if Path(save_path).is_file():
        return pickle.load(open(save_path, "rb"))
    else:
        parent = Path(save_path).parent.absolute()
        if not os.path.exists(parent):
            os.makedirs(parent)

        fake_sequences = []
        for start_idx, end_idx in tqdm(train_idx):
            curr_seq = train_data.iloc[start_idx:end_idx]["location_id"].values

            random_seq = curr_seq.copy()
            np.random.shuffle(random_seq)
            fake_sequences.append(random_seq)

            # random choose one location and switch to another location
            selected_idx = np.random.randint(len(curr_seq), size=1)
            curr_seq[selected_idx] = np.random.randint(len(all_locs) + 1, size=1)

            fake_sequences.append(curr_seq)

        save_pk_file(save_path, fake_sequences)

        return fake_sequences

def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

