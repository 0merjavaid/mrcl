import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import logging
import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearnerRegressionCol(nn.Module):
    def __init__(self, args, config, backbone_config=None, device='cpu'):
        """
        #
        :param args:
        """
        super(MetaLearnerRegressionCol, self).__init__()

        self.inner_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.TH = {}
        self.grads = {}
        self.TW = {}
        self.load_model(args, config, backbone_config, device)
        for named, param in self.named_parameters():
            if not param.meta:
                continue
            self.TH[named] = torch.zeros_like(param.data).to(device)
            self.grads[named] = torch.zeros_like(param.data).to(device)
            self.TW[named] = torch.zeros_like(param.data).to(device)

        forward_meta_weights = self.net.get_forward_meta_parameters()
        if len(forward_meta_weights) > 0:
            self.optimizer = optim.Adam(forward_meta_weights, lr=self.meta_lr)
        else:
            logger.warning("Zero meta parameters in the forward pass")

        # if args['model_path'] is not None:
        #     print (args['model_path'])
        #     self.load_weights(args)

        self.log_model()



    def log_model(self):
        for name, param in self.net.named_parameters():
            if param.meta:
                logger.info("Weight in meta-optimizer = %s %s", name, str(param.shape))
            if param.adaptation:
                logger.debug("Weight for adaptation = %s %s", name, str(param.shape))

    def load_model(self, args, config, context_config, device="cpu"):
        self.net = Learner.Learner(config, context_config, device=device)

    def update_TH(self, grads):
        with torch.no_grad():
            """
            hidden state not taken care of
            """
            counter = 0
            for name, a in self.named_parameters():
                if not a.meta:
                    continue
                if grads[counter] is not None:
                    if name in self.TH:
                        self.TH[name] = grads[counter]+ self.TH[name] * 0
                counter += 1

    def accumulate_gradients(self, target, y, hidden_state):
        with torch.no_grad():
            error = (target - y)
            prediction_params = self.net.get_adaptation_parameters()[-1]
            for name, param in self.named_parameters():
                if not param.meta:
                    continue
                if name in self.TH:
                    """
                    refactor
                    """
                    if self.TH[name].ndim == 3:
                        self.grads[name] = self.grads.get(name, 0) + (-1 * error * (prediction_params.view(1,1,-1) * self.TH[name] + self.TW[name]*hidden_state.view(1,1,-1)))
                    elif self.TH[name].ndim == 2:
                        self.grads[name] = self.grads.get(name, 0) + (-1 * error * (prediction_params.view(1,-1) * self.TH[name] + self.TW[name]*hidden_state.view(1,-1)))
                    elif self.TH[name].ndim == 1:
                        self.grads[name] = self.grads.get(name, 0) + (-1 * error * (prediction_params.view(-1) * self.TH[name] + self.TW[name]*hidden_state.view(-1)))
                    else:
                        0/0

    def update_TW(self, inner_lr, old_state, target, y):
        error = (target - y)
        with torch.no_grad():
            prediction_params = self.net.get_adaptation_parameters()[-1]
            for name, a in self.named_parameters():
                if not a.meta:
                    continue
                if name in self.TW:
                    # column_num = int(name.split(".")[1])
                    if self.TW[name].ndim == 3:
                        self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state.view(1,1,-1)*(prediction_params.view(1,1,-1)*self.TH[name] + old_state.view(1,1,-1)*self.TW[name])
                    elif self.TW[name].ndim == 2:
                        self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state.view(1,-1)*(prediction_params.view(1,-1)*self.TH[name] + old_state.view(1,-1)*self.TW[name])
                    elif self.TW[name].ndim == 1:
                        self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state.view(-1)*(prediction_params.view(-1)*self.TH[name] + old_state.view(-1)*self.TW[name])

                    else:
                        0/0

    def online_update(self, update_lr, rnn_state, target, y):
        error = (target - y)
        vars = self.net.get_adaptation_parameters()[0].clone() + update_lr * error * rnn_state.view(-1)
        self.net.update_weights([vars], meta=False)

    def reset_TH(self):
        for named, param in self.named_parameters():
            if not param.meta:
                continue
            self.TH[named] = torch.zeros_like(param.data)
            self.TW[named] = torch.zeros_like(param.data)

    def zero_grad(self):
        self.grads = {}
        for named, param in self.named_parameters():
            if not param.meta:
                continue
            self.grads[named] = torch.zeros_like(param.data)

    def reset_adaptation(self):
        self.net.reset_vars()

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        not doing 1st sample separatly
        """
        # meta loss
        total_loss = 0
        # inner updates in each step. no grad during prediction step bcz params being updated
        # shape of x_traj is 100x51. or 10x10x50. tasks x samples x 51
        # self.net is vectorized MetaLearnerFF
        self.reset_TH()
        self.zero_grad()
        self.reset_adaptation()
        self.optimizer.zero_grad()
        for k in range(0, len(x_traj)):
            value_prediction, rnn_state, grads = self.net.forward_col(x_traj[k], grad=True)
            # value_prediction, rnn_state, _ = self.net.forward_col(x_traj[k], grad=False)
            self.update_TH(grads)
            self.online_update(self.inner_lr, rnn_state, y_traj[k, 0], value_prediction)
            self.update_TW(self.inner_lr, rnn_state, y_traj[k, 0], value_prediction)

        for i in range(0, len(x_rand)):
            vp, rs, gs = self.net.forward_col(x_rand[i], grad=True)
            # vp, rs, _ = self.net.forward_col(x_rand[i], grad=False)
            self.update_TH(gs)
            total_loss += F.mse_loss(vp, y_rand[i, 0])
            self.accumulate_gradients(y_rand[i, 0], vp, hidden_state=rs)

        # grads = torch.autograd.grad(total_loss, self.net.get_forward_meta_parameters())
        # counter = 0
        # total_sum = 0
        # positive_sum = 0
        # dif = 0

        # for named, param in self.named_parameters():
        #     if not param.meta:
        #         continue
        #     dif += torch.abs(self.grads[named] - grads[counter]).sum()
        #     positive = ((self.grads[named] * grads[counter]) > 1e-10).float().sum()
        #     total = positive + ((self.grads[named] * grads[counter]) < - 1e-10).float().sum()
        #     total_sum += total
        #     positive_sum += positive
        #
        #     counter += 1
        #
        # logger.error("Difference = %s", (float(dif) / total_sum).item())
        # # gradient_error_list.append( (float(dif) / total_sum).item())
        # # gradient_alignment_list.append(str(float(positive_sum) / float(total_sum)))
        # logger.error("Grad alignment %s", str(float(positive_sum) / float(total_sum)))

        for name, p in self.named_parameters():
            if not p.meta:
                continue
            p.grad = self.grads[name].clone()
        self.optimizer.step()

        # self.net.prediction_params = self.net.init_param.clone()

        return 0, total_loss / len(x_rand)
