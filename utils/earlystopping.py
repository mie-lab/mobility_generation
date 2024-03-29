import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logdir, patience=7, verbose=False, delta=0, main_process=True, monitor="val_loss"):
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
        self.main_process = main_process
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.monitor = monitor
        self.best_return_dict = {self.monitor: np.inf}
        self.delta = delta
        self.save_name = None

    def __call__(self, return_dict, state_dict, save_name):
        score = return_dict[self.monitor]

        if self.best_score is None:
            self.best_score = score
            self.save_name = save_name
            self.save_checkpoint(return_dict, state_dict)
            return

        if score < self.best_score - self.delta:
            self.best_score = score
            self.save_name = save_name
            self.save_checkpoint(return_dict, state_dict)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.main_process:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                print(
                    f"{self.monitor} does not decrease: best {self.best_return_dict[self.monitor]:.6f} <--> now {return_dict[self.monitor]:.6f}."
                )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, return_dict, state_dict):
        """Saves model when self.monitor decrease."""
        if self.verbose and self.main_process:
            print(
                f"{self.monitor} decreased ({self.best_return_dict[self.monitor]:.6f} --> {return_dict[self.monitor]:.6f}).  Saving model ..."
            )
        if self.main_process:
            torch.save(state_dict, self.log_dir + f"/{self.save_name}.pt")
        self.best_return_dict = return_dict
