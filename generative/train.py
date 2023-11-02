import torch
import numpy as np
from tqdm import tqdm


def generate_samples(model, config):
    samples = []
    for _ in range(int(config.num_gen_samples / config.batch_size)):
        sample = model.sample(config.batch_size, config.generate_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)
