import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from networks import torch_utils, basic_networks
from networks.conv_lstm_cell import ConvLSTMCell
from tensorboardX import SummaryWriter


class KddDataset(Dataset):
    def __init__(self):
        # TODO xrz
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def random_split(dataset, lengths):
    lengths = [length * len(dataset) // sum(lengths) for length in lengths]
    lengths[-1] += len(dataset) - sum(lengths)
    indices = torch.randperm(len(dataset))
    return [Subset(dataset, indices[sum(lengths[:i]):sum(lengths[:i + 1])])
            for i in range(len(lengths))]


if __name__ == "__main__":
    # set manual seed
    torch.manual_seed(0)

    # define sequence_length, batch_size, channels, height, width
    T, B, C, H, W = 48, 64,  # TODO xrz should make these parames flexible in the dataset
    # For example, use past n days' data as input.
    HIDDEN_SIZE = 128
    LEARNING_RATE = 1e-4
    MAX_EPOCH = 400

    dataset = KddDataset(T, ...)
    train_set, validation_set, test_set = random_split(dataset, [2, 1, 1])
    train_loader = DataLoader(train_set, batch_size=B,
                              shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=B,
                                   shuffle=True)
    print('Size of train, validation, test: {}, {}, {}'.format(
        len(train_set), len(validation_set), len(test_set)))

    print('Instantiate model')
    aq_lstm = ...   # TODO
    me_lstm = ...
    grid_lstm = ConvLSTMCell(C, HIDDEN_SIZE)
    fusion = basic_networks.NNList([
        torch_utils.Flatten(),
        basic_networks.FC(
            input_dim=  # TODO concat output size of aq_lst, me_lstm and grid_lstm
            output_dim=  # TODO size of y
            hidden_dims=[HIDDEN_SIZE],
        ),
    ])

    print('Run on multiple gpus')
    aq_lstm = nn.DataParallel(aq_lstm).cuda()
    me_lstm = nn.DataParallel(me_lstm).cuda()
    grid_lstm = nn.DataParallel(grid_lstm).cuda()
    fusion = nn.DataParallel(fusion).cuda()
    models = [aq_lstm, me_lstm, grid_lstm, fusion]

    print('Create a criterion and optimizer')
    loss_fn = nn.XXXLoss()  # TODO find a proper loss for regression
    optimizer = optim.Adam([params for params in model.parameters()
                            for model in models], lr=LEARNING_RATE)

    print('Run for {MAX_EPOCH} epochs')
    writer = SummaryWriter()
    PLOT_EVERY = 10

    def run_batch(sample):
        aq, me, me_grid, y = sample[0].cuda(), sample[1].cuda(
        ), sample[2].cuda(), sample[3].cuda()

        # TODO run multiple modules on inputs
        # state = None
        # for t in range(T):
        # state = model(feature[:, t], state)

        # compute predicted value
        y_hat = fusion(...)
        loss = loss_fn(y_hat, y)

        smape =  # xrz can caculate our SMAPE score and
        # thus we can have a plot of it during training.
        # maybe we can find code on the forum
        return loss, smape

    for epoch in range(MAX_EPOCH):
        for sample_batched in train_loader:
            loss, smape = run_batch(sample_batched)

            print(' > Epoch {:2d} loss: {:.3f}'.format(epoch, loss.data[0]))

            # zero grad parameters
            for model in models:
                model.zero_grad()

            # compute new grad parameters through time!
            loss.backward()

            # learning_rate step against the gradient
            optimizer.step()

        if epoch % PLOT_EVERY == 0:
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('train_SMAPE',
                              misclassify, epoch)

            validation_loss, validation_misclassify = \
                run_batch(iter(validation_loader).next())
            writer.add_scalar('validation_loss',
                              validation_loss, epoch)
            writer.add_scalar('validation_SMAPE',
                              validation_misclassify, epoch)
