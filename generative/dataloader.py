import numpy as np
import pickle as pickle
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


import psutil
import datasets
from datasets import Dataset as Dataset2


import trackintel as ti


class discriminator_dataset(torch.utils.data.Dataset):
    def __init__(self, true_data, fake_data, valid_start_end_idx):
        super(discriminator_dataset, self).__init__()

        # reindex the df for efficient selection
        self.true_data = true_data.copy()
        self.true_data.set_index("id", inplace=True)

        self.fake_data = fake_data

        self.valid_start_end_idx = valid_start_end_idx
        self.true_len = len(valid_start_end_idx)
        self.fake_len = len(fake_data["locs"])

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.true_len + self.fake_len

    def __getitem__(self, idx):
        return_dict = {}

        if idx < self.true_len:
            start_end_idx = self.valid_start_end_idx[idx]
            selected = self.true_data.iloc[start_end_idx[0] : start_end_idx[1]]

            loc_seq = selected["location_id"].values[:-1]
            return_dict["duration"] = selected["act_duration"].values[:-1] // 30 + 1
            y = 1
        else:
            fake_idx = idx - self.true_len

            # :-1 already constructed in construct_discriminator_pretrain_dataset()
            loc_seq = self.fake_data["locs"][fake_idx]
            return_dict["duration"] = self.fake_data["durs"][fake_idx] // 30 + 1
            y = 0

        # [sequence_len]
        x = torch.tensor(loc_seq)
        return_dict["duration"] = torch.tensor(return_dict["duration"], dtype=torch.long)

        return x, y, return_dict


class generator_dataset(torch.utils.data.Dataset):
    def __init__(self, input_data, valid_start_end_idx):
        super(generator_dataset, self).__init__()

        self.data = input_data.copy()
        self.data.set_index("id", inplace=True)

        self.valid_start_end_idx = valid_start_end_idx
        self.len = len(valid_start_end_idx)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        start_end_idx = self.valid_start_end_idx[idx]
        selected = self.data.iloc[start_end_idx[0] : start_end_idx[1]]

        loc_seq = selected["location_id"].values
        # [sequence_len]
        x = torch.tensor(loc_seq[:-1])
        # [sequence_len]
        y = torch.tensor(loc_seq[1:])

        return_dict = {}
        y_dict = {}
        #
        return_dict["duration"] = torch.tensor(selected["act_duration"].values[:-1] // 30 + 1, dtype=torch.long)
        # predict without padding, need to add padding in autoregressive prediction
        y_dict["duration"] = torch.tensor(selected["act_duration"].values[1:] // 30, dtype=torch.float32)

        return x, y, return_dict, y_dict


def discriminator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    # get one sample batch
    src_dict_batch = {}
    for key in batch[0][-1]:
        src_dict_batch[key] = []

    for src_sample, tgt_sample, src_dict in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

        for key in src_dict:
            src_dict_batch[key].append(src_dict[key])

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    for key in src_dict_batch:
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0, batch_first=True)

    return src_batch, tgt_batch, src_dict_batch


def generator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    # get one sample batch
    src_dict_batch = {}
    for key in batch[0][-2]:
        src_dict_batch[key] = []

    tgt_dict_batch = {}
    for key in batch[0][-1]:
        tgt_dict_batch[key] = []

    for src_sample, tgt_sample, src_dict, tgt_dict in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

        for key in src_dict:
            src_dict_batch[key].append(src_dict[key])

        for key in tgt_dict:
            tgt_dict_batch[key].append(tgt_dict[key])

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    for key in src_dict_batch:
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0, batch_first=True)
    for key in tgt_dict_batch:
        tgt_dict_batch[key] = pad_sequence(tgt_dict_batch[key], padding_value=0, batch_first=True)

    return src_batch, tgt_batch, src_dict_batch, tgt_dict_batch


def generate_samples(model, seq_len, num, single_len=256, print_progress=False):
    samples = {"locs": [], "durs": []}
    for _ in tqdm(range(int(num / single_len)), disable=not print_progress):
        gen_samples = model.module.sample(single_len, seq_len)

        samples["locs"].extend(gen_samples["locs"].detach().cpu().numpy().tolist())
        samples["durs"].extend(gen_samples["durs"].detach().cpu().numpy().tolist())

    samples["locs"] = np.array(samples["locs"])
    samples["durs"] = np.array(samples["durs"])
    return samples


def get_pretrain_loaders(input_data, all_locs, world_size, config, device):
    train_data, train_idx, vali_data, vali_idx = input_data

    # train dataset
    fake_train_samples = _construct_discriminator_pretrain_dataset(train_data, train_idx, all_locs)
    fake_vali_samples = _construct_discriminator_pretrain_dataset(vali_data, vali_idx, all_locs)

    d_train_data = discriminator_dataset(
        true_data=train_data, fake_data=fake_train_samples, valid_start_end_idx=train_idx
    )
    train_sampler = DistributedSampler(d_train_data, num_replicas=world_size, rank=device, shuffle=True, drop_last=True)
    d_train_loader = torch.utils.data.DataLoader(
        d_train_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    # validation dataset
    d_vali_data = discriminator_dataset(true_data=vali_data, fake_data=fake_vali_samples, valid_start_end_idx=vali_idx)
    vali_sampler = DistributedSampler(d_vali_data, num_replicas=world_size, rank=device, shuffle=False)
    d_vali_loader = torch.utils.data.DataLoader(
        d_vali_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=vali_sampler,
    )

    # training dataset
    g_train_data = generator_dataset(input_data=train_data, valid_start_end_idx=train_idx)
    train_sampler = DistributedSampler(g_train_data, num_replicas=world_size, rank=device, shuffle=True)
    g_train_loader = torch.utils.data.DataLoader(
        g_train_data,
        collate_fn=generator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    # validation dataset
    g_vali_data = generator_dataset(input_data=vali_data, valid_start_end_idx=vali_idx)
    vali_sampler = DistributedSampler(g_vali_data, num_replicas=world_size, rank=device, shuffle=False)
    g_vali_loader = torch.utils.data.DataLoader(
        g_vali_data,
        collate_fn=generator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=vali_sampler,
    )

    if is_main_process():
        print(f"len d_train loader:\t{len(d_train_loader)}\t #samples: {len(d_train_data)}")
        print(f"len d_vali loader:\t{len(d_vali_loader)}\t #samples: {len(d_vali_data)}")
        print(f"len g_train loader:\t{len(g_train_loader)}\t #samples: {len(g_train_data)}")
        print(f"len g_vali loader:\t{len(g_vali_loader)}\t #samples: {len(g_vali_data)}")

    return d_train_loader, d_vali_loader, g_train_loader, g_vali_loader


def _construct_discriminator_pretrain_dataset(input_data, input_idx, all_locs):
    fake_seqs = {"locs": [], "durs": []}

    data_df = input_data.set_index("id")
    for start_idx, end_idx in tqdm(input_idx):
        curr_seq = data_df.iloc[start_idx:end_idx].copy()

        # only take the training idx
        loc_seq = curr_seq["location_id"].values[:-1]
        dur_seq = curr_seq["act_duration"].values[:-1]

        # random shuffle sequences; checked
        idx_arr = np.arange(len(loc_seq))
        np.random.shuffle(idx_arr)

        fake_seqs["locs"].append(loc_seq.copy()[idx_arr])
        fake_seqs["durs"].append(dur_seq.copy()[idx_arr])

        # random choose one location and switch to another location
        select_idx = np.random.randint(len(loc_seq), size=1)
        new_loc_seq = loc_seq.copy()
        new_dur_seq = dur_seq.copy()
        new_loc_seq[select_idx] = np.random.randint(low=1, high=len(all_locs) + 1, size=1)
        new_dur_seq[select_idx] = np.random.randint(low=0, high=60 * 24 * 2, size=1)

        fake_seqs["locs"].append(new_loc_seq)
        fake_seqs["durs"].append(new_dur_seq)

    return fake_seqs


def get_discriminator_dataloaders(train_data, train_idx, fake_data, world_size, config, device):
    d_train_data = discriminator_dataset(true_data=train_data, fake_data=fake_data, valid_start_end_idx=train_idx)
    train_sampler = DistributedSampler(d_train_data, num_replicas=world_size, rank=device, shuffle=True)
    d_train_loader = torch.utils.data.DataLoader(
        d_train_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    return d_train_data, d_train_loader


def load_data_text(
    batch_size,
    seq_len,
    deterministic=False,
    data_args=None,
    split="train",
    model_emb=None,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print("#" * 30, "\nLoading location data...")

    training_data = get_sequence(data_args, seq_len, split=split)

    dataset = DiffSeqDataset(training_data, data_args, model_emb=model_emb)

    if split != "test":
        sampler = DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            sampler=sampler,
            # shuffle=not deterministic,
            num_workers=0,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            # sampler=sampler,
            shuffle=not deterministic,
            num_workers=0,
        )

    return data_loader


def process_helper_fnc(seq_ls, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    seq_dataset = Dataset2.from_dict(seq_ls)
    # print(seq_dataset)
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(ls):
        lst = []
        mask = []
        for i in range(len(ls["src"])):
            src = ls["src"][i]
            tgt = ls["tgt"][i]

            lst.append(src + tgt)
            mask.append([0] * len(src))
        ls["input_ids"] = lst
        ls["input_mask"] = mask
        return ls

    seq_dataset = seq_dataset.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc="merge and mask",
    )

    def pad_function(ls):
        max_length = seq_len
        ls["input_ids"] = _collate_batch_helper(ls["input_ids"], 0, max_length)
        ls["input_mask"] = _collate_batch_helper(ls["input_mask"], 1, max_length)
        return ls

    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    seq_dataset = seq_dataset.map(
        pad_function,
        batched=True,
        num_proc=4,
        desc="padding",
        remove_columns=["src", "tgt"],
    )

    # print(seq_dataset, "padded dataset")
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets["train"] = seq_dataset
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_sequence(args, seq_len, split="train"):
    print("#" * 30, "\nLoading dataset {} from {}...".format(args.dataset, args.data_dir))

    print(f"### Loading form the {split} set...")
    path = (
        f"{args.data_dir}/{split}_level{args.level}_{args.src_min_days}_{args.tgt_min_days}_{args.dataset_variation}.pk"
    )

    sequence_ls = pickle.load(open(path, "rb"))

    processed_dict = {"src": [], "tgt": []}
    for record in sequence_ls:
        processed_dict["src"].append(record["src"])
        processed_dict["tgt"].append(record["tgt"])

    print("### Data samples...\n", processed_dict["src"][:1], processed_dict["tgt"][:1])

    train_dataset = process_helper_fnc(processed_dict, seq_len)
    return train_dataset


class DiffSeqDataset(torch.utils.data.Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets["train"])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr = self.text_datasets["train"][idx]["input_ids"]

        if self.model_emb is not None:
            with torch.no_grad():
                arr = self.model_emb(torch.tensor(arr))

        arr = np.array(arr, dtype=np.float32)
        out_kwargs = {}
        out_kwargs["input_ids"] = np.array(self.text_datasets["train"][idx]["input_ids"])
        out_kwargs["input_mask"] = np.array(self.text_datasets["train"][idx]["input_mask"])

        return arr, out_kwargs


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
