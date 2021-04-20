import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class MetaLearnerVecFF(nn.Module):
    def __init__(self, input_dimension, total_columns, column_width=10, total_neigbours=1, device="cpu",
                 targets_dim=1):
        super(MetaLearnerVecFF, self).__init__()

        self.fc1_weight = nn.Parameter(torch.zeros(input_dimension, column_width, total_columns))
        self.fc1_bias = nn.Parameter(torch.zeros(column_width, total_columns))

        self.fc2_weight = nn.Parameter(torch.zeros(column_width, column_width, total_columns))
        self.fc2_bias = nn.Parameter(torch.zeros(column_width, total_columns))

        self.i = nn.Parameter(torch.zeros(column_width, total_columns))
        self.i_bias = nn.Parameter(torch.zeros(total_columns))


        for named, param in self.named_parameters():
            torch.nn.init.uniform_(param, -1 * np.sqrt(1 / column_width), np.sqrt(1 / column_width))

        self.TH = {}
        self.grads = {}
        self.TW = {}

        for named, param in self.named_parameters():
            self.TH[named] = torch.zeros_like(param.data).to(device)
            self.grads[named] = torch.zeros_like(param.data).to(device)
            self.TW[named] = torch.zeros_like(param.data).to(device)

        self.prediction_params = torch.zeros(total_columns).to(device)
        # self.grads["prediction_params"] = torch.zeros_like(self.prediction_parameters.data).to(device)

    def forward(self, x, hidden_state, grad=True, retain_graph=False, bptt=False):

        if not bptt:
            hidden_state = nn.Parameter(hidden_state)

        x = x.view(-1, 1, 1)
        x = torch.relu(torch.sum(x * self.fc1_weight, 0) + self.fc1_bias) # x = 50x1x1 weight = 50x5x10
        # x = x.view(x.shape[0],1,x.shape[1])

        x = torch.relu(torch.sum(x * self.fc2_weight, 0) + self.fc2_bias) # x = 5x 10 weight = 5x5x10
        # x = x.view(x.shape[0], 1, x.shape[1])
        """
        No hidden state yet
        """
        h_t = torch.tanh(torch.sum(x * self.i, 0) + self.i_bias)
        sum_h = torch.sum(h_t)

        with torch.no_grad():
            grads = None
            if grad:
                """
                hidden state not taken care of
                """
                grads = torch.autograd.grad(sum_h, list(self.parameters()),
                                            allow_unused=True, retain_graph=retain_graph)

        y = torch.sum(self.prediction_params * h_t.view(-1))
        return y, h_t, grads

    def update_TH(self, grads):
        with torch.no_grad():
            """
            hidden state not taken care of
            """
            counter = 0
            for name, a in self.named_parameters():
                if grads[counter] is not None:
                    if name in self.TH:
                        self.TH[name] = grads[counter]+ self.TH[name] * 0
                counter += 1

    def accumulate_gradients(self, target, y, hidden_state):
        with torch.no_grad():
            error = (target - y)
            for name, param in self.named_parameters():
                if name in self.TH:
                    """
                    refactor
                    """
                    if self.TH[name].ndim == 3:
                        self.grads[name] = self.grads.get(name, 0) + (-1 * error * (self.prediction_params.view(1,1,-1) * self.TH[name] + self.TW[name]*hidden_state.view(1,1,-1)))
                    elif self.TH[name].ndim == 2:
                        self.grads[name] = self.grads.get(name, 0) + (-1 * error * (self.prediction_params.view(1,-1) * self.TH[name] + self.TW[name]*hidden_state.view(1,-1)))
                    elif self.TH[name].ndim == 1:
                        self.grads[name] = self.grads.get(name, 0) + (-1 * error * (self.prediction_params.view(-1) * self.TH[name] + self.TW[name]*hidden_state.view(-1)))
                    else:
                        0/0

    def update_TW(self, inner_lr, old_state, target, y):
        error = (target - y)
        with torch.no_grad():
            for name, a in self.named_parameters():
                if name in self.TW:
                    # column_num = int(name.split(".")[1])
                    if self.TW[name].ndim == 3:
                        self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state.view(1,1,-1)*(self.prediction_params.view(1,1,-1)*self.TH[name] + old_state.view(1,1,-1)*self.TW[name])
                    elif self.TW[name].ndim == 2:
                        self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state.view(1,-1)*(self.prediction_params.view(1,-1)*self.TH[name] + old_state.view(1,-1)*self.TW[name])
                    elif self.TW[name].ndim == 1:
                        self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state.view(-1)*(self.prediction_params.view(-1)*self.TH[name] + old_state.view(-1)*self.TW[name])

                    else:
                        0/0

    def online_update(self, update_lr, rnn_state, target, y):
        error = (target - y)
        self.prediction_params = self.prediction_params + update_lr * error * rnn_state.view(-1)

    def reset_TH(self):
        for named, param in self.named_parameters():
            if "weight" in named or "bias" in named:
                self.TH[named] = torch.zeros_like(param.data)
                self.TW[named] = torch.zeros_like(param.data)


class MetaLearnerRegressionCol(nn.Module):
    def __init__(self, input_dimension, total_columns, column_width=10, device="cpu",
                inner_lr=0.001, meta_lr=0.001):
        super(MetaLearnerRegressionCol, self).__init__()
        self.net = MetaLearnerVecFF(input_dimension, total_columns, column_width, 0, device=device).to(device)
        self.rnn_state = torch.zeros(total_columns)
        self.net.reset_TH()
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        not doing 1st sample separatly
        """
        # meta loss
        total_loss = 0
        # inner updates in each step. no grad during prediction step bcz params being updated
        # shape of x_traj is 100x51. or 10x10x50. tasks x samples x 51
        # self.net is vectorized MetaLearnerFF
        for k in range(0, len(x_traj)):
            value_prediction, self.rnn_state, grads = self.net.forward(x_traj[k], self.rnn_state, grad=True, bptt=False)
            # internally self.prediction_params are being maintained
            total_loss += F.mse_loss(value_prediction, y_traj[k, 0])
            self.net.update_TH(grads)
            self.net.accumulate_gradients(y_traj[k, 0], value_prediction, hidden_state=self.rnn_state)
            self.net.online_update(self.inner_lr, self.rnn_state, y_traj[k, 0], value_prediction)
            self.net.update_TW(self.inner_lr, self.rnn_state, y_traj[k, 0], value_prediction)

        # for meta update. shape here 100x51 or 10x10x51
        # for i in range(0, len(x_rand)):
        #     value_prediction, self.rnn_state, grads = self.net(x_rand[i], self.rnn_state, grad=True)
        #     total_loss += F.mse_loss(value_prediction, y_rand[k, 0])
        #     self.net.accumulate_gradients(y_rand[i, 0], value_prediction, hidden_state=self.rnn_state)

        self.optimizer.zero_grad()
        self.optimizer.step()

        return 0, total_loss / len(x_rand)
