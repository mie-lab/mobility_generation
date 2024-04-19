import torch


# creating the dataloader
class pretrain_dataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        super(pretrain_dataset, self).__init__()

        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
