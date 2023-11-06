import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logdir, patience=7, verbose=False, delta=0, monitor="val_loss", save_name="checkpoint"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.log_dir = logdir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.monitor = monitor
        self.best_return_dict = {self.monitor: np.inf}
        self.delta = delta
        self.save_name = save_name

    def __call__(self, return_dict, model):
        score = return_dict[self.monitor]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            return

        if score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, return_dict, model):
        """Saves model when self.monitor decrease."""
        if self.verbose:
            print(
                f"{self.monitor} decreased ({self.best_return_dict[self.monitor]:.6f} --> {return_dict[self.monitor]:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.log_dir + f"/{self.save_name}.pt")
        self.best_return_dict = return_dict
