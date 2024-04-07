import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

import time
import blobfile as bf

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils.earlystopping import EarlyStopping
from utils.dist_util import load_state_dict


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

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=1e-5)

    return optim


def train_net(config, model, train_loader, val_loader, device, log_dir):
    performance = {}

    optim = get_optimizer(config, model)

    # define learning rate schedule
    if config.decay_epochs == 0:
        scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps=len(train_loader) * config.warmup_epochs)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=len(train_loader) * config.warmup_epochs,
            num_training_steps=len(train_loader) * config.decay_epochs,
        )

    scheduler_ES = StepLR(optim, step_size=1, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        log_dir,
        patience=config["patience"],
        main_process=True,
        verbose=config.verbose,
        monitor="val_loss",
        delta=0.0001,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_fp16)

    # Loop for n_epochs
    for epoch in range(1000):
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
            scaler,
            globaliter,
        )

        # At the end of the epoch, do a pass on the validation set
        return_dict = single_validate(config, model, val_loader, device)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "lr_schedule": scheduler_ES.state_dict(),
        }

        early_stopping(return_dict, checkpoint, save_name=f"model_{epoch}")

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
            checkpoint = load_state_dict(bf.join(log_dir, early_stopping.save_name + ".pt"))
            model.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            scheduler_ES.load_state_dict(checkpoint["lr_schedule"])

            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
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
    scaler,
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
    for i, inputs in enumerate(train_loader):
        optim.zero_grad()
        globaliter += 1

        x, y, x_dict, y_dict = send_to_device(inputs, device, config)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.use_fp16):
            logits, dur_pred = model(x, x_dict)

            loc_loss_size = CEL(logits, y.reshape(-1))
            dur_loss_size = MSE(dur_pred.reshape(-1), y_dict["duration"].reshape(-1))
            loss = loc_loss_size + config.loss_weight * dur_loss_size / (dur_loss_size / loc_loss_size).detach()

        scaler.scale(loss).backward()

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optim)
        scaler.update()

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

            logits, dur_pred = model(x, x_dict)

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
            x, y, x_dict, _ = send_to_device(inputs, device, config)

            logits, _ = model(x, x_dict)

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
    generated_dict = {"pred": []}
    with torch.no_grad():
        user_ls = []
        count = 0
        for inputs in tqdm(data_loader):
            x, _, x_dict, _ = send_to_device(inputs, device, config)

            len_before = x_dict["len"].detach().clone()
            for _ in range(config.generate_len):
                logits, dur_pred = model(x, x_dict)

                pred_loc = top_k_top_p_filtering(logits, top_k=0, top_p=0.99, filter_value=-float("Inf"))

                # append to the end of sequence for next prediction
                x = torch.stack(
                    [
                        torch.cat([xi[:x_leni], pred_loci, xi[x_leni:]])
                        for xi, x_leni, pred_loci in zip(x, x_dict["len"], pred_loc)
                    ]
                )
                x_dict["len"] = x_dict["len"] + 1

            len_after = x_dict["len"].detach().clone()
            # collect the simulated sequences, first dim is the batch dim
            generated_dict["pred"].append(
                torch.stack(
                    [xi[len_beforei:len_afteri] for xi, len_beforei, len_afteri in zip(x, len_before, len_after)]
                )
            )

            user_ls.append(x_dict["user"])

            count = count + 1
            if config.debug and count == 20:
                break

    generated_dict["pred"] = torch.cat(generated_dict["pred"], dim=0).cpu().numpy().astype(int)
    user_arr = torch.cat(user_ls, dim=0).cpu().numpy()

    if config.verbose:
        print(len(generated_dict["pred"]), user_arr.shape)

    return generated_dict, user_arr


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return pred_token
