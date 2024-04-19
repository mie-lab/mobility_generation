import argparse

import pandas as pd
import geopandas as gpd
import datetime
from shapely import wkt
import numpy as np
import torch

from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split

from easydict import EasyDict as edict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import setup_seed, load_config, init_save_path
from pretrain.dataloader import pretrain_dataset
from pretrain.model import SkipGram, NegativeSamplingLoss

import shapely

from scipy.spatial import KDTree
import pickle as pickle


def get_all_coordinates():
    # get all possible locations
    all_locs = pd.read_csv("./data/s2_loc_visited_level10_13.csv", index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    all_locs = all_locs.to_crs("EPSG:2056")

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        all_locs["loc_id"].values.reshape(-1, 1)
    )
    # add 1 to account for 0 padding
    all_locs["loc_id"] = enc.transform(all_locs["loc_id"].values.reshape(-1, 1)) + 2

    print(
        f"Max loc id {all_locs.loc_id.max()}, min loc id {all_locs.loc_id.min()}, unique loc id:{all_locs.loc_id.unique().shape[0]}"
    )

    return shapely.get_coordinates(all_locs.geometry)


def single_train(
    log_interval,
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    globaliter,
):
    model.train()

    running_loss = 0.0

    n_batches = len(loader)
    start_time = time.time()
    for i, inputs in enumerate(loader):
        globaliter += 1
        source, targets, negative = inputs

        source = torch.LongTensor(source).to(device)  # [b]
        targets = torch.LongTensor(targets).to(device)  # [b]
        negative = negative.long().to(device)  # [b*config.negative_num]

        embedded_inputs = model.forward_input(source)
        embedded_targets = model.forward_target(targets)
        embedded_noise = model.forward_noise(negative)

        loss = criterion(embedded_inputs, embedded_targets, embedded_noise)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % log_interval == 0:
            print(
                "Epoch {}, {:.1f}%\t loss: {:.4f} , took: {:.2f}s \r".format(
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / log_interval,
                    time.time() - start_time,
                ),
                end="",
                flush=True,
            )

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
    print()
    return globaliter


def single_validate(model, loader, criterion, device):
    total_val_loss = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in loader:
            source, targets, negative = inputs

            source = torch.LongTensor(source).to(device)  # [b]
            targets = torch.LongTensor(targets).to(device)  # [b]
            negative = negative.long().to(device)  # [b*config.negative_num]

            embedded_inputs = model.forward_input(source)
            embedded_targets = model.forward_target(targets)
            embedded_noise = model.forward_noise(negative)

            loss = criterion(embedded_inputs, embedded_targets, embedded_noise)
            total_val_loss += loss.item()

    val_loss = total_val_loss / len(loader)

    print("Validation loss = {:.2f}".format(val_loss))
    return {"val_loss": val_loss}


def consine_validation(model, loader, device):
    # validation
    close = []
    far = []
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(loader):
            source, targets, negative = inputs

            source = torch.LongTensor(source).to(device)  # [b]
            targets = torch.LongTensor(targets).to(device)  # [b]
            negative = negative.long().to(device)  # [b*config.negative_num]

            embedded_inputs = model.forward_input(source)
            embedded_targets = model.forward_target(targets)
            embedded_noise = model.forward_noise(negative)

            embedded_inputs = embedded_inputs.cpu().numpy()
            embedded_targets = embedded_targets.cpu().numpy()
            embedded_noise = embedded_noise.cpu().numpy()

            batch_close = cosine_similarity(embedded_inputs, embedded_targets).diagonal()
            close.extend(batch_close)

            batch_far = [
                np.mean(cosine_similarity(input.reshape(1, -1), noise))
                for input, noise in zip(embedded_inputs, embedded_noise)
            ]
            far.extend(batch_far)

    print("Close distance: {:.2f} Noise distance: {:.2f}".format(np.mean(close), np.mean(far)))


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./pretrain/skip-gram.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    setup_seed(config.seed)
    timestamp_now = int(datetime.datetime.now().timestamp())

    coordinates = get_all_coordinates()

    tree = KDTree(coordinates)

    # for point in coordinates:
    _, idx = tree.query(coordinates, k=config.neighbors + 1, workers=1)
    positive = idx[:, 1:]

    pairs = []
    for i, ls in tqdm(enumerate(positive)):
        ls = ls + 2
        i = i + 2

        remain = np.setdiff1d(np.arange(2, config.max_location), ls)
        for cand in ls:
            negative = np.random.choice(a=remain, size=config.negative_num)

            pairs.append([i, cand, negative])

    print(len(pairs))

    # pairs_train, pairs_test = train_test_split(pairs, test_size=0.2, random_state=config.seed)

    # train
    train_dataset = pretrain_dataset(pairs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    # validate
    # vali_dataset = pretrain_dataset(pairs_test)
    # vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkipGram(max_location=config.max_location, hidden_dim=config.hidden_dim).to(device)
    criterion = NegativeSamplingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    globaliter = 0
    training_start_time = time.time()

    for epoch in range(20):
        single_train(
            config.log_interval,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            globaliter,
        )

        # At the end of the epoch, do a pass on the validation set
        # return_dict = single_validate(model, vali_loader, criterion, device)

        # consine_validation(model, vali_loader, device)

        print("=" * 50)
    consine_validation(model, train_loader, device)

    embeddings = model.in_embed.weight.to("cpu").data.numpy()
    save_pk_file("./data/matrix/loc_embedding.pk", embeddings)
