import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner

logger = logging.getLogger("experiment")


class RTRLMetaRegression(nn.Module):
    def __init__(self, args, config, backbone_config=None, device="cpu"):
        """
        #
        :param args:
        """
        super(RTRLMetaRegression, self).__init__()
        self.inner_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.static_plasticity = args["static_plasticity"]
        self.context_plasticity = args["context_plasticity"]

        self.sigmoid = not args["no_sigmoid"]
        self.load_model(args, config, backbone_config, device)
        self.optimizers = []

        forward_meta_weights = self.net.get_forward_meta_parameters()
        if len(forward_meta_weights) > 0:
            self.optimizer_forward_meta = optim.Adam(forward_meta_weights, lr=self.meta_lr)
            self.optimizers.append(self.optimizer_forward_meta)
        else:
            logger.warning("Zero meta parameters in the forward pass")

        if args["static_plasticity"]:
            self.net.add_static_plasticity()
            self.optimizer_static_plasticity = optim.Adam(self.net.static_plasticity, lr=args["plasticity_lr"])
            self.optimizers.append(self.optimizer_static_plasticity)

        if args["neuro"]:
            self.net.add_neuromodulation(args['context_dimension'])
            self.optimizer_neuro = optim.Adam(self.net.neuromodulation_parameters, lr=args["neuro_lr"])
            self.optimizers.append(self.optimizer_neuro)

        if args['model_path']:
            self.load_weights(args)

        self.log_model()

    def log_model(self):
        for name, param in self.net.named_parameters():
            print(name)
            if param.meta:
                logger.info("Weight in meta-optimizer = %s %s", name, str(param.shape))
            if param.adaptation:
                logger.debug("Weight for adaptation = %s %s", name, str(param.shape))

    def optimizers_zero_grad(self):
        for opti in self.optimizers:
            opti.zero_grad()

    def optimizer_step(self):
        for opti in self.optimizers:
            opti.step()

    def load_model(self, args, config, context_config, device="cpu"):
        if args['model_path'] is not None and False:
            pass
            assert (False)
        else:
            self.net = Learner.Learner(config, context_config, type="representation", device=device)

    def load_weights(self, args):
        pass

    def online_update(self, update_lr, representation, target, weight):
        y = torch.sum(weight.view(-1)*representation.view(-1))
        error = (target - y)
        new_weights = weight.view(-1) + update_lr * error * representation.view(-1)
        return new_weights


    def reset_adaptation(self):
        self.net.reset_vars()

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        total_loss = 0
        self.reset_adaptation()
        self.optimizers_zero_grad()
        # inner updates RTRL
        old_weights = torch.nn.Parameter(self.net.get_adaptation_parameters()[0])
        jacobians_t = []
        representation = self.net.forward(x_traj[0], rep=True)
        fast_weights = self.online_update(self.inner_lr, representation, y_traj[0, 0], old_weights)
        jacobian_w = torch.autograd.grad(fast_weights, [old_weights], grad_outputs=torch.ones_like(fast_weights),
                                         allow_unused=False)
        # Update the d w /  d (\theta) matrix here
        assert(False)
        for k in range(1, len(x_traj)):

            representation = self.net.forward(x_traj[k], rep=True)
            old_weights = fast_weights
            fast_weights =  self.online_update(self.inner_lr, representation, y_traj[k, 0], old_weights)
            jacobian_w = torch.autograd.grad(fast_weights, [old_weights], grad_outputs=torch.ones_like(fast_weights), allow_unused=False)
            # Update the d w /  d (\theta) matrix here
            assert(False)
            """
            Wrong equations...
            """
            # d(w_t-1) / d(theta)
            jacobian_theta = torch.autograd.grad(w_prev, self.net.get_forward_meta_parameters().clone())
            
            jacobian = jacobian_theta + torch.matmul(jacobian_w, jacobian_theta)
            jacobians_t.append(jacobian)


        for i in range(0, len(x_rand)):
            vp, rs, _ = self.net.forward_rtrl(x_rand[i])
            total_loss += F.mse_loss(vp, y_rand[i, 0])

        dW = torch.autograd.grad(total_loss, w_t)
        d_theta = torch.matmul(dW, jacobian).squeeze(0)
        ind = 0
        # copy params to network params
        for name, p in self.named_parameters():
            if not p.meta:
                continue
            p.grad = d_theta[ind].clone()
            ind += 1

        assert len(d_theta) == ind
        self.optimizer_step()
        return 0, total_loss / len(x_rand)


class MetaLearnerRegression(nn.Module):

    def __init__(self, args, config, backbone_config=None, device="cpu"):
        """
        #
        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

        self.inner_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.static_plasticity = args["static_plasticity"]
        self.context_plasticity = args["context_plasticity"]

        self.sigmoid = not args["no_sigmoid"]
        self.load_model(args, config, backbone_config, device)
        self.optimizers = []

        forward_meta_weights = self.net.get_forward_meta_parameters()
        if len(forward_meta_weights) > 0:
            self.optimizer_forward_meta = optim.Adam(forward_meta_weights, lr=self.meta_lr)
            self.optimizers.append(self.optimizer_forward_meta)
        else:
            logger.warning("Zero meta parameters in the forward pass")

        if args["static_plasticity"]:
            self.net.add_static_plasticity()
            self.optimizer_static_plasticity = optim.Adam(self.net.static_plasticity, lr=args["plasticity_lr"])
            self.optimizers.append(self.optimizer_static_plasticity)

        if args["neuro"]:
            self.net.add_neuromodulation(args['context_dimension'])
            self.optimizer_neuro = optim.Adam(self.net.neuromodulation_parameters, lr=args["neuro_lr"])
            self.optimizers.append(self.optimizer_neuro)

        if args['model_path'] is not "None":
            self.load_weights(args)

        self.log_model()

    def log_model(self):
        for name, param in self.net.named_parameters():
            print(name)
            if param.meta:
                logger.info("Weight in meta-optimizer = %s %s", name, str(param.shape))
            if param.adaptation:
                logger.debug("Weight for adaptation = %s %s", name, str(param.shape))

    def optimizer_zero_grad(self):
        for opti in self.optimizers:
            opti.zero_grad()

    def optimizer_step(self):
        for opti in self.optimizers:
            opti.step()

    def load_model(self, args, config, context_config, device="cpu"):
        if args['model_path'] is not None and False:
            pass
            assert (False)
        else:
            self.net = Learner.Learner(config, context_config, device=device)

    def load_weights(self, args):
        loaded_net = torch.load(args['model_path'] + "/net.model",
                                map_location="cpu")

        for (n1, old_model), (n2, loaded_model) in zip(loaded_net.named_parameters(), self.net.named_parameters()):
            if n1 == n2:
                loaded_model.data = old_model.data
            else:
                print(n1, n2)
                assert (False)

    def online_update(self, update_lr, rnn_state, target, y):
        error = (target - y)
        vars = self.net.get_adaptation_parameters()[0].clone() + update_lr * error * rnn_state.view(-1)
        self.net.update_weights([vars], meta=False)

    def inner_update(self, net, vars, grad, adaptation_lr, list_of_context=None, log=False):
        adaptation_weight_counter = 0

        new_weights = []
        for p in vars:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                if self.context_plasticity:
                    g = g * list_of_context[adaptation_weight_counter].view(g.shape)
                if self.static_plasticity:
                    mask = net.static_plasticity[adaptation_weight_counter].view(g.shape)
                    g = g * torch.sigmoid(mask)

                temp_weight = p - adaptation_lr * g
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

    def clip_grad(self, grad, norm=10):
        grad_clipped = []
        for g, p in zip(grad, self.net.parameters()):
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            grad_clipped.append(g)
        return grad_clipped

    def clip_grad_inplace(self, net, norm=10):
        for p in net.parameters():
            g = p.grad
            p.grad = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = p.grad
            p.grad = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
        return net

    def reset_adaptation(self):
        self.net.reset_vars()

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        not doing 1st sample separatly
        """
        # meta loss
        total_loss = 0
        self.reset_adaptation()
        self.optimizer_forward_meta.zero_grad()
        for k in range(0, len(x_traj)):
            value_prediction, rnn_state, _ = self.net.forward_col(x_traj[k], grad=False)
            self.online_update(self.inner_lr, rnn_state, y_traj[k, 0], value_prediction)

        for i in range(0, len(x_rand)):
            # vp, rs, gs = self.net.forward_col(x_rand[i], grad=True)
            vp, rs, _ = self.net.forward_col(x_rand[i], grad=False)
            # self.update_TH(gs)
            total_loss += F.mse_loss(vp, y_rand[i, 0])
            # self.accumulate_gradients(y_rand[i, 0], vp, hidden_state=rs)
        self.optimizer_forward_meta.zero_grad()
        grads = torch.autograd.grad(total_loss, self.net.get_forward_meta_parameters())
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

        ind = 0
        for name, p in self.named_parameters():
            if not p.meta:
                continue
            p.grad = grads[ind].clone()
            ind+=1

        assert len(grads) == ind

        self.optimizer_forward_meta.step()

        return 0, total_loss / len(x_rand)


    def forward_(self, x_traj, y_traj, x_rand, y_rand):
        prediction, _, _ = self.net.forward_col(x_traj[0], vars=None, grad=False)
        loss = F.mse_loss(prediction, y_traj[0, 0])
        grad = self.clip_grad(torch.autograd.grad(loss, self.net.get_adaptation_parameters(),
                                                  create_graph=True))
        list_of_context = None
        if self.context_plasticity:
            list_of_context = self.net.forward_plasticity(x_traj[0])

        fast_weights = self.inner_update(self.net, self.net.parameters(), grad, self.update_lr, list_of_context)

        with torch.no_grad():
            prediction, _, _ = self.net.forward_col(x_rand[0], vars=None, grad=False)
            first_loss = F.mse_loss(prediction, y_rand[0, 0])
        for k in range(1, len(x_traj)):
            prediction, _, _ = self.net.forward_col(x_traj[k], fast_weights, grad=False)
            loss = F.mse_loss(prediction, y_traj[k, 0])
            grad = self.clip_grad(torch.autograd.grad(loss, self.net.get_adaptation_parameters(fast_weights),
                                                      create_graph=True))

            list_of_context = None
            if self.context_plasticity:
                list_of_context = self.net.forward_plasticity(x_traj[k])
            fast_weights = self.inner_update(self.net, fast_weights, grad, self.update_lr, list_of_context)
        final_meta_loss = 0
        for i in range(1, len(x_rand)):
            prediction_qry_set, _, _ = self.net.forward_col(x_rand[i], fast_weights, grad=False)
            final_meta_loss += F.mse_loss(prediction_qry_set, y_rand[i, 0])

        self.optimizer_zero_grad()

        final_meta_loss.backward()

        self.optimizer_step()
        return [first_loss.detach(), final_meta_loss.detach()/(len(x_rand) - 1)]


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args['update_lr']
        self.meta_lr = args['meta_lr']
        self.update_step = args['update_step']

        self.net = Learner.Learner(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight)

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        assert(steps < 16)
        counter = 0
        #
        x_rand_temp = []
        y_rand_temp = []

        class_counter = 0
        for it1 in iterators:
            # assert (len(iterators) == 1)
            steps_inner = 0
            rand_counter = 0
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)

                counter += 1
                if steps_inner < steps:
                    x_traj.append(img)
                    y_traj.append(data)
                    steps_inner += 1

                else:
                    x_rand_temp.append(img)
                    y_rand_temp.append(data)
                    rand_counter += 1
                    if rand_counter == 5:
                        break
            class_counter += 1

        # Sampling the random batch of data
        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)


        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        # print(y_traj)
        # print(y_rand)
        #
        # quit()
        return x_traj, y_traj, x_rand, y_rand

    def sample_training_data_paper(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        assert(steps < 16)
        counter = 0
        #
        x_rand_temp = []
        y_rand_temp = []

        class_counter = 0
        for it1 in iterators:
            # assert (len(iterators) == 1)
            steps_inner = 0
            rand_counter = 0
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)

                counter += 1
                if steps_inner < steps:
                    x_traj.append(img)
                    y_traj.append(data)
                    steps_inner += 1

                else:
                    x_rand_temp.append(img)
                    y_rand_temp.append(data)
                    rand_counter += 1
                    if rand_counter == steps:
                        break
            class_counter += 1

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)

        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), x_rand_temp, y_rand_temp


        return x_traj, y_traj, x_rand, y_rand

    def inner_update(self, x, fast_weights, y):
        adaptation_weight_counter = 0

        logits = self.net(x, fast_weights)
        loss = F.cross_entropy(logits, y)
        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = torch.autograd.grad(loss, self.net.get_adaptation_parameters(fast_weights),
                                   create_graph=True)

        new_weights = []
        for p in fast_weights:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                temp_weight = p - self.update_lr * g
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

    def meta_loss(self, x, fast_weights, y):

        logits = self.net(x, fast_weights)
        loss_q = F.cross_entropy(logits, y)
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """

        meta_losses = [0 for _ in range(len(x_traj) + 1)]  # losses_q[i] is the loss on step i
        accuracy_meta_set = [0 for _ in range(len(x_traj) + 1)]

        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(x_traj[0], None, y_traj[0])

        with torch.no_grad():
            # Meta loss before any inner updates
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], self.net.parameters(), y_rand[0])
            meta_losses[0] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy

            # Meta loss after a single inner update
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0])
            meta_losses[1] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy

        for k in range(1, len(x_traj)):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k])

            # Computing meta-loss with respect to latest weights
            meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0])
            meta_losses[k + 1] += meta_loss

            # Computing accuracy on the meta and traj set for understanding the learning
            with torch.no_grad():
                pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
                accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy

        # Taking the meta gradient step
        self.optimizer.zero_grad()
        meta_loss = meta_losses[-1]
        meta_loss.backward()

        self.optimizer.step()
        accuracies = np.array(accuracy_meta_set) / len(x_rand[0])
        print (accuracies[-1], meta_losses)
        return accuracies, meta_losses


def main():
    pass


if __name__ == '__main__':
    main()
