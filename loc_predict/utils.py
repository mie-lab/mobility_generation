import os

import pickle as pickle
import pandas as pd
import datetime
import json

from loc_predict.train import train_net, single_test, get_performance_dict, generate
from loc_predict.models import TransEncoder, RNNs
from loc_predict.models.markov import generate_markov


def get_trained_nets(config, model, train_loader, val_loader, device, log_dir):
    best_model, perf = train_net(config, model, train_loader, val_loader, device, log_dir=log_dir)
    perf["type"] = "vali"
    return best_model, perf


def get_test_result(config, best_model, test_loader, device):
    return_dict = single_test(config, best_model, test_loader, device)
    performance = get_performance_dict(return_dict)
    performance["type"] = "test"

    return performance


def get_models(config, device):
    if config.networkName == "mhsa":
        model = TransEncoder(config=config).to(device)
    elif config.networkName == "rnn":
        model = RNNs(config=config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of trainable parameters: ", total_params)

    return model


def get_generated_sequences(config, model, test_loader, device=None):
    if config.networkName == "mhsa":
        generated_ls, user_arr = generate(config, model, test_loader, device)
    else:
        generated_ls, user_arr = generate_markov(config, model, test_loader)

    generated_df = pd.DataFrame([user_arr, generated_ls])
    generated_df = generated_df.transpose()
    generated_df.columns = ["user_id", "generated_ls"]

    generated_df = generated_df.explode(column=["generated_ls"])
    generated_df.index.name = "seq_id"

    return generated_df
