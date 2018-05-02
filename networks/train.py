import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from networks import torch_utils, basic_networks
from networks.conv_lstm_cell import ConvLSTMCell
from utils.dataset import KddDataset, random_split
from utils.eval import SMAPE


if __name__ == "__main__":
    # set manual seed
    torch.manual_seed(0)

    # define sequence_length, batch_size, channels, height, width
    T, T_PRED, B, C, H, W, AQ_SIZE = 48, 24, 16, 5, 31, 21, 105
    HIDDEN_SIZE = 32
    NUM_LAYERS = 1
    LEARNING_RATE = 1e-4
    MAX_EPOCH = 400

    dataset = torch.load('data/dataset_bj.pt')
    train_set, validation_set, test_set = random_split(dataset, [3, 1, 1])
    train_loader = DataLoader(train_set, batch_size=B, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=B, shuffle=True)
    print('Size of train, validation, test: {}, {}, {}'.format(
        len(train_set), len(validation_set), len(test_set)))

    print('Instantiate model')
    aq_lstm = nn.LSTM(input_size=AQ_SIZE, hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS, batch_first=True)
    me_lstm = ConvLSTMCell(C, HIDDEN_SIZE)
    fusion = basic_networks.NNList([
        torch_utils.Flatten(),
        basic_networks.FC(
            input_dim=T * HIDDEN_SIZE + HIDDEN_SIZE * H * W,
            output_dim=T_PRED * AQ_SIZE,
            hidden_dims=[HIDDEN_SIZE],
        ),
    ])

    print('Run on multiple gpus')
    aq_lstm = aq_lstm.cuda()
    me_lstm = nn.DataParallel(me_lstm).cuda()
    fusion = fusion.cuda()
    models = [aq_lstm, me_lstm, fusion]

    print('Create a criterion and optimizer')
    loss_fn = nn.MSELoss(reduce=False)
    optimizer = optim.Adam(
        [params for model in models for params in model.parameters()], lr=LEARNING_RATE)

    print('Run for {MAX_EPOCH} epochs')
    writer = SummaryWriter()
    PLOT_EVERY = 1

    def run_batch(sample):
        B = sample[0].shape[0]
        aq, me, me_pred, y = sample[0].cuda(), sample[1].cuda(
        ), sample[2].cuda(), sample[3].cuda()
        me = torch.cat([me, me_pred], dim=1)

        hidden = (Variable(torch.zeros(NUM_LAYERS, B, HIDDEN_SIZE)).cuda(),
                  Variable(torch.zeros(NUM_LAYERS, B, HIDDEN_SIZE)).cuda())
        aq_state = aq_lstm(aq, hidden)
        aq_feature = aq_state[0].contiguous().view(B, -1)

        me_state = None
        for t in range(T):
            me_state = me_lstm(me[:, t], me_state)
        me_feature = me_state[0].view(B, -1)

        # compute predicted value
        y_hat = fusion(torch.cat([aq_feature, me_feature], dim=1)).view(B, T_PRED, AQ_SIZE)
        loss = loss_fn(y_hat, y)
        mask = torch.isnan(loss)
        loss[mask] = 0
        loss = torch.sum(loss) / int(np.prod(loss.shape) - torch.sum(mask))

        smape = SMAPE(y.data, y_hat.data)
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
            for model in models:
                for param in model.parameters():
                    param.grad[torch.isnan(param.grad)] = 0

            # learning_rate step against the gradient
            optimizer.step()

        if epoch % PLOT_EVERY == 0:
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('train_SMAPE', smape, epoch)

            validation_loss, validation_smape = \
                run_batch(iter(validation_loader).next())
            print(' > Epoch {:2d} validation loss: {:.3f}'.format(epoch, validation_loss.data[0]))
            writer.add_scalar('validation_loss',
                              validation_loss, epoch)
            writer.add_scalar('validation_SMAPE',
                              validation_smape, epoch)
