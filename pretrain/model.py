import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, max_location, hidden_dim):
        super(SkipGram, self).__init__()

        self.in_embed = nn.Embedding(max_location, hidden_dim)
        self.out_embed = nn.Embedding(max_location, hidden_dim)

        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-0.8, 0.8)
        self.out_embed.weight.data.uniform_(-0.8, 0.8)

    def forward_input(self, x):
        return self.in_embed(x)

    def forward_target(self, x):
        return self.out_embed(x)

    def forward_noise(self, x):
        """Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        # use context matrix for embedding noise samples
        return self.out_embed(x)


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape

        input_vectors = input_vectors.view(batch_size, embed_size, 1)  # batch of column vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)  # batch of row vectors

        # log-sigmoid loss for correct pairs
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()

        # log-sigmoid loss for incorrect pairs
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        return -(out_loss + noise_loss).mean()  # average batch loss
