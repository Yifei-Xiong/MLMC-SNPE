import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from sbibm.utils.nflows import get_flow


class Cond_NSF(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_hidden, device):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_layer: layer size of Block
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        """

        super(Cond_NSF, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.nsf_from_sbibm = get_flow(model="nsf", dim_distribution=dim_out, dim_context=dim_in-dim_out,
                                       hidden_features=n_hidden[0], flow_num_transforms=n_layer).to(device)

    def log_density_value_at_data(self, data_sample, theta_sample):
        return self.nsf_from_sbibm.log_prob(theta_sample, data_sample)

    def gen_sample(self, sample_size, x_0):
        return self.nsf_from_sbibm.sample(int(sample_size), x_0).squeeze(0)


class DefaultModelParam:
    def __init__(self):
        self.n_layer = 8  # 8
        self.batch_norm = False
        self.round = 20  # 20
        self.round_sample_size = 1000
        self.valid_rate = 0.05
        self.valid_interval = 10
        self.medd_samp_size = 2000
        self.medd_round = 1  # 10
        self.mmd_samp_size = 2000
        self.mmd_round = 1
        self.steps = 10000
        self.print_state = self.steps
        self.print_state_time = 10000
        self.batch_size = 100
        self.meshgrid_plot_point = 10
        self.figure_dpi = 400
        self.detected_log_file = False  # True: if log file exist, exit the program
        self.plot_loss_figure_save = True  # True
        self.plot_mmd_figure_save = False  # False
        self.plot_theta_figure_save = True  # True
        self.plot_density_figure_show = False  # False
        self.plot_density_figure_save = False  # False
        self.save_theta_csv = True  # True
        self.clear_cuda_cache = True
        self.manual_seed = None
        self.show_detailed_info = False
        self.linux_path = None
        self.calc_c2st = True
