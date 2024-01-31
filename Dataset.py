# load dataset for SNPE model

import torch
import math
import ctypes
import time
import os
import numpy as np
import pandas as pd
import SNPE_lib
from torch.multiprocessing import Pool

torch.multiprocessing.set_sharing_strategy('file_system')


class Simulator:
    def __init__(self, dataset_name, device, dtype=torch.float32, normalize=False):
        if os.sep == "/":
            self.FileSavePath = ""  # put your path here
        else:
            self.FileSavePath = ""  # put your path here
        self.dataset_name = dataset_name
        self.dim_theta = 0
        self.dim_x = 0
        self.device = device
        self.cache_theta = None
        self.dtype = dtype
        self.can_sample_from_post = False
        self.bounded_prior = False
        self.true_theta = None
        self.normalize = normalize
        assert dataset_name in ['two_moons', 'lotka', 'mg1']
        if dataset_name == 'two_moons':
            self.dim_x = 2
            self.dim_theta = 2
            self.x_0 = torch.tensor([[0., 0.]], device=device)
            self.unif_lower = torch.tensor([-1., -1.], device=self.device)
            self.unif_upper = -self.unif_lower
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.can_sample_from_post = True
            self.bounded_prior = True
            self.scale = torch.tensor([1.00, 1.00], device=self.device)
            self.true_theta = torch.tensor([[0.2475, 0.2475]], device=torch.device('cpu'))
            self.columns = ['$\\theta_1$', '$\\theta_2$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "reference_theta" + os.sep + "Da+0.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'lotka':
            self.dim_x = 9
            self.dim_theta = 4
            self.unif_lower = torch.tensor([-5., -5., -5., -5.], device=self.device)
            self.unif_upper = torch.tensor([2., 2., 2., 2.], device=self.device)
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.x_0 = torch.tensor([[4.6431, 4.0170, 7.1992, 6.6024, 0.9765, 0.9237, 0.9712, 0.9078, 0.047567]], device=self.device)
            self.scale = torch.tensor([0.3294, 0.5483, 0.6285, 0.9639, 0.0091, 0.0222, 0.0107, 0.0224, 0.1823], device=self.device)
            self.bounded_prior = True
            self.true_theta = torch.log(torch.tensor([[0.01, 0.5, 1, 0.01]], device=torch.device('cpu')))
            self.columns = ['$\\theta_1$', '$\\theta_2$', '$\\theta_3$', '$\\theta_4$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "reference_theta" + os.sep + "Da+1.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'mg1':
            self.dim_x = 5
            self.dim_theta = 3
            self.unif_lower = torch.tensor([0., 0., 0.], device=self.device)
            self.unif_upper = torch.tensor([10., 10., 1 / 3], device=self.device)
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.x_0 = torch.log(torch.tensor([[1.0973, 2.3010, 4.2565, 7.2229, 23.3592]], device=self.device))
            self.scale = torch.tensor([0.1049, 0.1336, 0.1006, 0.1893, 0.2918], device=self.device)
            self.bounded_prior = True
            self.true_theta = torch.tensor([[1., 4., 0.2]], device=torch.device('cpu'))
            self.columns = ['$\\theta_1$', '$\\theta_2$', '$\\theta_3$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "reference_theta" + os.sep + "Da+2.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        else:
            raise NotImplementedError
        self._x_0 = torch.clone(self.x_0)

    def gen_data(self, para, param=None):
        # input param: theta
        # input shape: batch * dim_theta
        # output param: x from p(x|theta)
        # output shape: batch * dim_x
        batch = para.shape[0]
        if self.dataset_name == 'two_moons':
            unif_dist = torch.distributions.Uniform(- math.pi / 2, math.pi / 2)
            normal_dist = torch.distributions.Normal(0.1, 0.01)  # mu, sigma
            a_sample = unif_dist.sample((batch,)).to(self.device)  # batch * 1
            r_sample = normal_dist.sample((batch,)).to(self.device)  # batch * 1
            dim_1 = r_sample * torch.cos(a_sample) + 0.25 - (torch.abs(para[:, 0] + para[:, 1])) / torch.sqrt(torch.tensor(2.))
            dim_2 = r_sample * torch.sin(a_sample) + (-para[:, 0] + para[:, 1]) / torch.sqrt(torch.tensor(2.))
            return torch.stack((dim_1, dim_2), dim=1)
        elif self.dataset_name == 'lotka':
            # multiprocess, calculate by C++ shared library
            cpudv = torch.device('cpu')
            para_c = torch.exp(para.cpu().type(torch.float64))
            rand_int = torch.randint(65536, size=(batch,)).cpu().reshape(-1, 1)
            para_c = torch.cat((para_c, rand_int), dim=1)
            if os.sep == "\\":
                Cfun = ctypes.WinDLL(self.FileSavePath + 'liblotka_c.dll', winmode=0)
            else:
                Cfun = ctypes.CDLL(self.FileSavePath + 'liblotka_c.so')
            n = self.dim_x  # length for each task
            s = 15 if os.sep == "\\" else 14  # number of threads
            k = batch  # number of tasks
            input_value = torch.cat((para_c, torch.zeros(batch, n - 5).cpu()), dim=1)
            output_value = torch.zeros(input_value.shape[0], input_value.shape[1], dtype=self.dtype, device=cpudv)
            num_parts = (batch + k - 1) // k
            for i in range(num_parts):
                start_idx = i * k
                end_idx = min((i + 1) * k, batch)
                input_list = [float(s), float(k)] + input_value[start_idx:end_idx].reshape(-1).tolist()
                c_values = (ctypes.c_double * len(input_list))(*input_list)
                Cfun.lotka_multi_thread(c_values)
                output_value[start_idx:end_idx] = torch.tensor([c_values[j + 2] for j in range(len(c_values) - 2)], device=cpudv).reshape(-1, n)
            output_value[:, 0:2] = torch.log(output_value[:, 0:2] + 1.)
            return output_value.to(self.device)
        elif self.dataset_name == 'mg1':
            job_num = 50
            quantile = torch.tensor([0., 0.25, 0.50, 0.75, 1.], device=self.device)
            zero_tensor = torch.tensor(0., device=self.device)
            unif_dist = torch.distributions.Uniform(0, 1)
            serv_time = para[:, 0].view(-1, 1) + unif_dist.sample((batch, job_num)).to(self.device) * para[:, 1].view(-1, 1)
            inter_time = -torch.log(unif_dist.sample((batch, job_num)).to(self.device) + 1e-8) / para[:, 2].view(-1, 1)
            arr_time = torch.cumsum(inter_time, dim=1)
            inter_left_time = torch.zeros(batch, job_num, device=self.device)
            left_time = torch.zeros(batch, job_num, device=self.device)
            inter_left_time[:, 0] = serv_time[:, 0] + arr_time[:, 0]
            left_time[:, 0] = inter_left_time[:, 0]
            for i in range(1, job_num):
                inter_left_time[:, i] = serv_time[:, i] + torch.max(zero_tensor, arr_time[:, i] - left_time[:, i - 1])
                left_time[:, i] = left_time[:, i - 1] + inter_left_time[:, i]
            return torch.log(torch.nanquantile(inter_left_time, quantile, dim=1).t())
        else:
            raise NotImplementedError
