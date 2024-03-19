import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from tqdm import tqdm

import time

from transformers import get_linear_schedule_with_warmup

from utils.earlystopping import EarlyStopping


def send_to_device(inputs, device, config):
    x, y, x_dict, y_dict = inputs
    if config.networkName == "deepmove":
        x = (x[0].to(device), x[1].to(device))

        for key in x_dict[0]:
            x_dict[0][key] = x_dict[0][key].to(device)
        for key in x_dict[1]:
            x_dict[1][key] = x_dict[1][key].to(device)
    else:
        x = x.to(device)
        for key in x_dict:
            x_dict[key] = x_dict[key].to(device)
        for key in y_dict:
            y_dict[key] = y_dict[key].to(device)
    y = y.to(device)

    return x, y, x_dict, y_dict


def get_performance_dict(return_dict):
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "f1": return_dict["f1"] * 100,
        "total": return_dict["total"],
    }

    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100

    return perf


def calculate_correct_total_prediction(logits, true_y):
    top1 = []
    result_ls = []
    for k in [1, 3, 5, 10]:
        if logits.shape[-1] < k:
            k = logits.shape[-1]

        prediction = torch.topk(logits, k=k, dim=-1).indices

        # f1 score
        if k == 1:
            top1 = torch.squeeze(prediction, dim=-1).cpu()

        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        # top_k = np.sum([curr_y in pred for pred, curr_y in zip(prediction, true_y)])
        result_ls.append(top_k)

    # mrr
    result_ls.append(get_mrr(logits, true_y))
    # total
    result_ls.append(true_y.shape[0])

    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1


def get_mrr(prediction, targets):
    """
    Calculates the MRR score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)

    return torch.sum(rranks).cpu().numpy()


def get_optimizer(config, model):
    # define the optimizer & learning rate
    if config.optimizer == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True,
        )
    elif config.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    return optim


def train_net(config, model, train_loader, val_loader, device, log_dir):
    performance = {}

    optim = get_optimizer(config, model)

    # define learning rate schedule
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optim, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config["patience"], verbose=config.verbose, delta=0.001)

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        # train for one epoch
        globaliter = single_train(
            config,
            model,
            train_loader,
            optim,
            device,
            epoch,
            scheduler,
            scheduler_count,
            globaliter,
        )

        # At the end of the epoch, do a pass on the validation set
        return_dict = single_validate(config, model, val_loader, device)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
            if scheduler_count == 2:
                performance = get_performance_dict(early_stopping.best_return_dict)
                print(
                    "Training finished.\t Time: {:.2f}min.\t validation acc@1: {:.2f}%".format(
                        (time.time() - training_start_time) / 60,
                        performance["acc@1"],
                    )
                )

                break

            scheduler_count += 1
            model.load_state_dict(torch.load(log_dir + "/checkpoint.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
            # print("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
            # print("Current learning rate: {:.5f}".format(scheduler_ES.get_last_lr()[0]))
            print("Current learning rate: {:.5f}".format(optim.param_groups[0]["lr"]))
            print("=" * 50)

        if config.debug is True:
            break

    return model, performance


def single_train(
    config,
    model,
    train_loader,
    optim,
    device,
    epoch,
    scheduler,
    scheduler_count,
    globaliter,
):
    model.train()

    running_loss = 0.0
    # 1, 3, 5, 10, rr, total
    result_arr = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    n_batches = len(train_loader)

    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    MSE = torch.nn.MSELoss(reduction="mean")

    # define start time
    start_time = time.time()
    optim.zero_grad()
    for i, inputs in enumerate(train_loader):
        globaliter += 1

        x, y, x_dict, y_dict = send_to_device(inputs, device, config)

        logits, dur_pred = model(x, x_dict, device)

        loc_loss_size = CEL(logits, y.reshape(-1))
        dur_loss_size = MSE(dur_pred.reshape(-1), y_dict["duration"].reshape(-1))
        loss = loc_loss_size + config.loss_weight * dur_loss_size / (dur_loss_size / loc_loss_size).detach()

        optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        if scheduler_count == 0:
            scheduler.step()

        # Print statistics
        running_loss += loss.item()

        batch_result_arr, _, _ = calculate_correct_total_prediction(logits, y)
        result_arr += batch_result_arr

        if (config.verbose) and ((i + 1) % config["print_step"] == 0):
            print(
                "Epoch {}, {:.1f}%\t loss: {:.3f} acc@1: {:.2f} mrr: {:.2f}, took: {:.2f}s \r".format(
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / config["print_step"],
                    100 * result_arr[0] / result_arr[-1],
                    100 * result_arr[4] / result_arr[-1],
                    time.time() - start_time,
                ),
                end="",
                flush=True,
            )

            # Reset running loss and time
            running_loss = 0.0
            result_arr = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            start_time = time.time()

        if config["debug"] and (i > 20):
            break
    if config.verbose:
        print()
    return globaliter


def single_validate(config, model, data_loader, device):
    total_val_loss = 0
    true_ls = []
    top1_ls = []

    result_arr = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    MSE = torch.nn.MSELoss(reduction="mean")
    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x, y, x_dict, y_dict = send_to_device(inputs, device, config)

            logits, dur_pred = model(x, x_dict, device)

            loc_loss_size = CEL(logits, y.reshape(-1))
            dur_loss_size = MSE(dur_pred.reshape(-1), y_dict["duration"].reshape(-1))
            loss = loc_loss_size + config.loss_weight * dur_loss_size / (dur_loss_size / loc_loss_size).detach()

            total_val_loss += loss.item()

            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(logits, y)
            result_arr += batch_result_arr

            true_ls.extend(batch_true.tolist())
            if not batch_top1.shape:
                top1_ls.extend([batch_top1.tolist()])
            else:
                top1_ls.extend(batch_top1.tolist())

    val_loss = total_val_loss / len(data_loader)

    f1 = f1_score(true_ls, top1_ls, average="weighted")

    if config.verbose:
        print(
            "Validation loss = {:.2f} acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                val_loss,
                100 * result_arr[0] / result_arr[-1],
                100 * f1,
                100 * result_arr[4] / result_arr[-1],
            ),
        )

    return {
        "val_loss": val_loss,
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": f1,
        "rr": result_arr[4],
        "total": result_arr[5],
    }


def single_test(config, model, data_loader, device):
    true_ls = []
    top1_ls = []

    result_arr = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x, y, x_dict, y_dict = send_to_device(inputs, device, config)

            logits, dur_pred = model(x, x_dict, device)

            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(logits, y)
            result_arr += batch_result_arr
            true_ls.extend(batch_true.numpy())
            top1_ls.extend(batch_top1.numpy())

    # f1 score
    f1 = f1_score(true_ls, top1_ls, average="weighted")

    if config.verbose:
        print(
            "Test acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                100 * result_arr[0] / result_arr[-1],
                100 * f1,
                100 * result_arr[4] / result_arr[-1],
            ),
        )

    return {
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": f1,
        "rr": result_arr[4],
        "total": result_arr[5],
    }


def generate(config, model, data_loader, device):
    model.eval()
    with torch.no_grad():
        generated_ls = []
        user_ls = []
        count = 0
        for inputs in tqdm(data_loader):
            x, _, x_dict = send_to_device(inputs, device, config)

            len_before = x_dict["len"].detach().clone()
            for _ in range(config.generate_len):
                logits = model(x, x_dict, device)

                # TODO: implement greedy, sample and beam search
                top = torch.topk(logits, k=10, dim=-1)

                p = torch.cumsum(top.values / top.values.sum(dim=-1, keepdim=True), dim=-1)

                idx = torch.searchsorted(p, torch.rand([p.shape[0], 1]).to(device))
                pred_loc = top.indices.gather(dim=1, index=idx)

                # append to the end of sequence for next prediction
                x = torch.stack(
                    [
                        torch.cat([xi[:x_leni], pred_loci, xi[x_leni:]])
                        for xi, x_leni, pred_loci in zip(x.transpose(1, 0), x_dict["len"], pred_loc)
                    ]
                ).transpose(1, 0)
                x_dict["len"] = x_dict["len"] + 1

            len_after = x_dict["len"].detach().clone()
            # collect the simulated sequences, first dim is the batch dim
            generated_ls.append(
                torch.stack(
                    [
                        xi[len_beforei:len_afteri]
                        for xi, len_beforei, len_afteri in zip(x.transpose(1, 0), len_before, len_after)
                    ]
                )
            )

            user_ls.append(x_dict["user"])

            count = count + 1
            if config.debug and count == 20:
                break

    generated_ls = torch.cat(generated_ls, dim=0).cpu().numpy().tolist()
    user_arr = torch.cat(user_ls, dim=0).cpu().numpy()

    if config.verbose:
        print(len(generated_ls), user_arr.shape)

    return generated_ls, user_arr
