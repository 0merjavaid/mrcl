import logging

logger = logging.getLogger("experiment")

import logging

import torch
from torch import nn
from torch.nn import functional as F
import oml

logger = logging.getLogger("experiment")


class Learner(nn.Module):
    """
    """

    def __init__(self, learner_configuration, backbone_configuration=None, type="columnar", device="cpu"):
        """

        :param learner_configuration: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = learner_configuration
        self.backbone_config = backbone_configuration

        self.vars = nn.ParameterList()
        if type == "columnar":
            self.vars = self.parse_config_columnar(self.config, nn.ParameterList())
        else:
            self.vars = self.parse_config(self.config, nn.ParameterList())

        self.vars = self.vars.to(device)
        self.adapt_layer = self.get_init_adap()
        self.context_backbone = None

    def parse_config(self, config, vars_list):
        for i, info_dict in enumerate(config):
            if info_dict["name"] == 'conv2d':
                w, b = oml.nn.conv2d(info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                w, b = oml.nn.linear(param_config["out"], param_config["in"], info_dict["adaptation"],
                                     info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] in ['tanh', 'rep', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'rotate']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list

    def parse_config_columnar(self, config, vars_list):
        for i, info_dict in enumerate(config):
            if info_dict["name"] in ['linear', 'hidden']:
                param_config = info_dict["config"]
                params = oml.nn.col_linear(param_config["cols"], param_config["out"], param_config["in"], info_dict["adaptation"],
                                     info_dict["meta"], info_dict['bias'])
                for p in params:
                    vars_list.append(p)

            elif info_dict["name"] in ['tanh', 'rep', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'rotate']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list

    def add_rotation(self):
        self.rotate = nn.Parameter(torch.ones(2304,2304))
        torch.nn.init.uniform_(self.rotate)
        self.rotate_inverse = nn.Parameter(torch.inverse(self.rotate))
        # print(self.rotate.shape)
        # print(self.rotate_inverse.shape)
        # quit()
        logger.info("Inverse computed")

    def get_init_adap(self):
        vars = []
        for var in self.vars:
            if var.adaptation:
                vars.append(var.clone())
        return vars

    def reset_vars(self):
        """
        Reset all adaptation parameters to random values. Bias terms are set to zero and other terms to default values of kaiming_normal_
        :return:
        """
        i = 0
        for var in self.vars:
            if var.adaptation is True:
                var.data = self.adapt_layer[i].data
                i += 1
                # if len(var.shape) > 1:
                #     torch.nn.init.kaiming_normal_(var)
                # else:
                #     torch.nn.init.zeros_(var)

    def forward_col(self, x, vars=None, grad=True, config=None, retain_graph=False):
        x = x.float()
        h_t = None
        grads = None
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0
        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]
            if name == 'linear':
                w = vars[idx]
                b = 0 if idx+1 == len(config) else vars[idx+1]
                if x.ndim == 1:
                    x = x.view(-1, 1, 1)
                elif x.ndim == 2 and w.ndim == 3:
                    x = x.unsqueeze(1)
                elif info_dict.get('adaptation'):
                    # prediction layer
                    h_t = x.view(-1)
                    sum_ht = torch.sum(h_t)
                    w = w.view(-1)
                    y = torch.sum(h_t * w, 0) + b
                    idx = idx+1 if isinstance(b, int) else idx+2
                    continue
                x = torch.sum(x * w, 0) + b
                idx += 2
            elif name == 'hidden':
                w, b = vars[idx], vars[idx + 1]
                w = w.squeeze(1)
                x = torch.sum(x * w, 0) + b
                idx += 2
            elif name == 'relu':
                x = F.relu(x)
            else:
                raise NotImplementedError
        if grad:
            with torch.no_grad():
                grads = torch.autograd.grad(sum_ht, self.get_forward_meta_parameters(),
                                            allow_unused=True, retain_graph=retain_graph)
        assert idx == len(vars)
        return y, h_t, grads

    def forward_rtrl(self, x, vars=None, config=None, retain_graph=False):
        x = x.float()
        h_t = None

        if vars is None:
            vars = self.vars
        if config is None:
            config = self.config

        idx = 0
        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]
            if name == 'linear':
                w = vars[idx]
                b = 0 if idx+1 == len(vars) else vars[idx+1]

                if info_dict.get('adaptation'):
                    # prediction layer No bias
                    h_t = x.view(-1)
                    y = F.linear(x, w)
                    idx = idx+1 if isinstance(b, int) else idx+2
                    continue
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'hidden':
                w, b = vars[idx], vars[idx + 1]
                w = w.squeeze(1)
                x = torch.sum(x * w, 0) + b
                idx += 2
            elif name == 'relu':
                x = F.relu(x)
            else:
                raise NotImplementedError

        assert idx == len(vars)
        return y, h_t, None

    def forward(self, x, vars=None, config=None, sparsity_log=False, rep=False):
        """
        """
        x = x.float()
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0

        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=info_dict['config']['stride'], padding=info_dict['config']['padding'])
                idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'rotate':
                # pass
                x = F.linear(x, self.rotate)
                x = F.linear(x, self.rotate_inverse)

            elif name == 'reshape':
                continue

            elif name == 'rep':
                if rep:
                    return x

            elif name == 'relu':
                x = F.relu(x)

            else:
                raise NotImplementedError
        assert idx == len(vars)
        return x

    def update_weights(self, vars, meta=False):
        i = 0
        for old in self.vars:
            if old.meta == meta:
                old.data = vars[i].data
                i += 1
                # don't update bias
                break
        assert i == len(vars)

    def get_adaptation_parameters(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: x.adaptation, list(vars)))

    def get_forward_meta_parameters(self):
        """
        :return: meta parameters i.e. parameters changed in the meta update
        """
        return list(filter(lambda x: x.meta, list(self.vars)))

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """

        return self.vars
