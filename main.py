import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import math
import pandas as pd
import argparse
import os
import shutil
import copy
import Dataset
import SNPE_lib
import seaborn as sns
import scipy
from sbibm import metrics


def atom_loss(batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density):
    current_batch_size = batch_data_sample.shape[0]
    loss_part_1 = 0.
    loss_part_2 = 0.
    loss_part_3 = 0.
    l_list_int = torch.zeros(current_batch_size, dtype=torch.int)
    size_firs = current_batch_size
    m_list_int = M0 * torch.pow(2, l_list_int)
    inner_theta_sample_size = torch.sum(m_list_int)
    # generate inner theta sample
    inner_theta_idx = torch.multinomial(torch.ones(current_batch_size), inner_theta_sample_size, replacement=True)
    inner_theta_all = batch_theta_sample[inner_theta_idx]
    inner_idx_cumsum = torch.cumsum(m_list_int, dim=0)
    inner_theta_all[torch.cat((torch.tensor([0]), inner_idx_cumsum[:(current_batch_size - 1)]))] = batch_theta_sample
    inner_theta_all[inner_idx_cumsum - 1] = batch_theta_sample
    loss_part_1 -= torch.sum(density_family.log_density_value_at_data(batch_data_sample[:size_firs], batch_theta_sample[:size_firs])
                             + batch_theta_log_density - batch_prior_logprob)
    data_firs_expand = batch_data_sample[:size_firs].view(size_firs, 1, -1).expand(size_firs, M0, -1).reshape(
        size_firs * M0, -1)
    loss_part_2 += torch.sum(torch.logsumexp(
        density_family.log_density_value_at_data(data_firs_expand, inner_theta_all).view(size_firs, M0) -
        prior.log_prob(inner_theta_all).view(size_firs, M0), dim=1) - np.log(M0))
    return loss_part_1, loss_part_2, loss_part_3, current_batch_size


def nested_loss(batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density):
    current_batch_size = batch_data_sample.shape[0]
    loss_part_1 = 0.
    loss_part_2 = 0.
    loss_part_3 = 0.
    l_list_int = torch.zeros(current_batch_size, dtype=torch.int)
    size_firs = current_batch_size
    m_list_int = M0 * torch.pow(2, l_list_int)
    inner_theta_sample_size = torch.sum(m_list_int)
    # generate inner theta sample
    inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), inner_theta_sample_size, replacement=True)
    inner_theta_all = theta_sample[inner_theta_idx]
    inner_idx_cumsum = torch.cumsum(m_list_int, dim=0)
    inner_theta_all[torch.cat((torch.tensor([0]), inner_idx_cumsum[:(current_batch_size - 1)]))] = batch_theta_sample
    inner_theta_all[inner_idx_cumsum - 1] = batch_theta_sample
    loss_part_1 -= torch.sum(density_family.log_density_value_at_data(batch_data_sample[:size_firs], batch_theta_sample[:size_firs])
                             + batch_theta_log_density - batch_prior_logprob)
    data_firs_expand = batch_data_sample[:size_firs].view(size_firs, 1, -1).expand(size_firs, M0, -1).reshape(
        size_firs * M0, -1)
    loss_part_2 += torch.sum(torch.logsumexp(
        density_family.log_density_value_at_data(data_firs_expand, inner_theta_all).view(size_firs, M0) -
        prior.log_prob(inner_theta_all).view(size_firs, M0), dim=1) - np.log(M0))
    return loss_part_1, loss_part_2, loss_part_3, current_batch_size


def ru_mlmc_loss(batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density, L_cut: int = 12):
    # L_cut should be set to a large number (like 12) to prevent out of VRAM
    current_batch_size = batch_data_sample.shape[0]
    loss_part_1 = 0.
    loss_part_2 = 0.
    loss_part_3 = 0.
    l_list_double, _ = torch.sort(geom_dist.sample((current_batch_size,)))
    l_list_int = l_list_double.int()
    l_list_int[l_list_int > L_cut] = L_cut
    size_firs = torch.sum(l_list_int == 0)
    m_list_int = M0 * torch.pow(2, l_list_int)
    inner_theta_sample_size = torch.sum(m_list_int)
    # generate inner theta sample
    inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), inner_theta_sample_size, replacement=True)
    inner_theta_all = theta_sample[inner_theta_idx]
    inner_idx_cumsum = torch.cumsum(m_list_int, dim=0)
    inner_theta_all[torch.cat((torch.tensor([0]), inner_idx_cumsum[:(current_batch_size - 1)]))] = batch_theta_sample
    inner_theta_firs = inner_theta_all[:(size_firs * M0)]
    loss_part_1 -= torch.sum(density_family.log_density_value_at_data(batch_data_sample[:size_firs], batch_theta_sample[:size_firs])
                             + batch_theta_log_density[:size_firs] - batch_prior_logprob[:size_firs]) / mlmc_w0
    data_firs_expand = batch_data_sample[:size_firs].view(size_firs, 1, -1).expand(size_firs, M0, -1).reshape(
        size_firs * M0, -1)
    loss_part_2 += torch.sum(torch.logsumexp(
        density_family.log_density_value_at_data(data_firs_expand, inner_theta_firs).view(size_firs, M0) -
        prior.log_prob(inner_theta_firs).view(size_firs, M0), dim=1) - np.log(M0)) / mlmc_w0
    data_seco_expand = torch.repeat_interleave(batch_data_sample[size_firs:].cpu(), m_list_int[size_firs:], dim=0).to(device)
    if data_seco_expand.shape[0] != 0:
        inner_theta_all[inner_idx_cumsum - 1] = batch_theta_sample
        inner_theta_seco = inner_theta_all[(size_firs * M0):]
        inner_logp = density_family.log_density_value_at_data(data_seco_expand, inner_theta_seco) - prior.log_prob(inner_theta_seco)
        start_idx = 0
        for length in l_list_int[size_firs:].tolist():
            end_idx = start_idx + M0 * pow(2, length)
            mid_idx = (start_idx + end_idx) >> 1
            loss_part_3 += (torch.logsumexp(inner_logp[start_idx:end_idx], dim=0) -
                            0.5 * torch.logsumexp(inner_logp[start_idx:mid_idx], dim=0) -
                            0.5 * torch.logsumexp(inner_logp[mid_idx:end_idx], dim=0) - np.log(2)) / (mlmc_w0 * pow(2, -mlmc_alpha * length))
            start_idx = end_idx
    return loss_part_1, loss_part_2, loss_part_3, current_batch_size


def grr_mlmc_loss(batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density, L_cut: int = 8, M_base: int = 2):
    # GRR loss
    current_batch_size = batch_data_sample.shape[0]
    loss_part_1 = 0.
    loss_part_2 = 0.
    loss_part_3 = 0.
    l_list_double, sort_idx = torch.sort(geom_dist.sample((current_batch_size,)))
    sort_data_sample = batch_data_sample[sort_idx]
    sort_theta_sample = batch_theta_sample[sort_idx]
    sort_prior_logprob = batch_prior_logprob[sort_idx]
    sort_theta_log_density = batch_theta_log_density[sort_idx]
    l_list_int = l_list_double.int()
    l_list_int[l_list_int > L_cut] = L_cut
    l_list_int[l_list_int < M_base] = M_base
    max_L = torch.max(l_list_int)
    for l_idx in range(M_base, max_L+1):  # l_idx = M_base, M_base+1, ..., torch.max(l_list_int).
        nl = torch.sum(l_list_int >= l_idx)  # numbers of samples that should be calculated at level l_idx
        Ml = M0 * pow(2, l_idx)
        if l_idx == M_base:
            # in this case, nl == current_batch_size, Ml == M0 * 2^M_base
            inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), nl * Ml, replacement=True)
            inner_theta_all = theta_sample[inner_theta_idx]
            inner_theta_all[torch.linspace(0, (nl - 1) * Ml, nl.item(), dtype=torch.int32)] = sort_theta_sample
            loss_part_1 -= torch.sum(density_family.log_density_value_at_data(sort_data_sample, sort_theta_sample)
                                     + sort_theta_log_density - sort_prior_logprob)
            data_firs_expand = sort_data_sample.view(current_batch_size, 1, -1).expand(current_batch_size, Ml, -1).reshape(
                current_batch_size * Ml, -1)
            loss_part_2 += torch.sum(torch.logsumexp(
                density_family.log_density_value_at_data(data_firs_expand, inner_theta_all).view(current_batch_size, Ml) -
                prior.log_prob(inner_theta_all).view(current_batch_size, Ml), dim=1) - np.log(Ml))
        else:
            inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), nl * Ml, replacement=True)
            inner_theta_all = theta_sample[inner_theta_idx]
            inner_theta_all[torch.linspace(0, (nl - 1) * Ml, nl.item(), dtype=torch.int32)] = sort_theta_sample[:nl]
            inner_theta_all[torch.linspace(Ml - 1, nl * Ml - 1, nl.item(), dtype=torch.int32)] = sort_theta_sample[:nl]
            data_firs_expand = sort_data_sample[:nl].view(nl, 1, -1).expand(nl, Ml, -1).reshape(nl * Ml, -1)
            inner_logp = density_family.log_density_value_at_data(data_firs_expand, inner_theta_all) - prior.log_prob(inner_theta_all)
            loss_part_3 += (torch.sum(torch.logsumexp(inner_logp.view(nl, Ml), dim=1) - np.log(2)) -
                            torch.sum(torch.logsumexp(inner_logp.view(nl << 1, Ml >> 1), dim=1)) / 2) / pow(2, -mlmc_alpha * l_idx)
    return loss_part_1, loss_part_2, loss_part_3, current_batch_size


def tgrr_mlmc_loss(batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density, L: int = 4, M_base: int = 2):
    # TGRR loss
    current_batch_size = batch_data_sample.shape[0]
    loss_part_1 = 0.
    loss_part_2 = 0.
    loss_part_3 = 0.
    calib_weight = torch.pow(torch.tensor(2.), -mlmc_alpha * torch.linspace(0, L, L + 1))
    calib_weight /= torch.sum(calib_weight)
    l_list_int, sort_idx = torch.sort(torch.multinomial(calib_weight, current_batch_size, replacement=True))
    sort_data_sample = batch_data_sample[sort_idx]
    sort_theta_sample = batch_theta_sample[sort_idx]
    sort_prior_logprob = batch_prior_logprob[sort_idx]
    sort_theta_log_density = batch_theta_log_density[sort_idx]
    l_list_int[l_list_int < M_base] = M_base
    max_L = torch.max(l_list_int)
    for l_idx in range(M_base, max_L+1):  # l_idx = M_base, M_base+1, ..., torch.max(l_list_int).
        nl = torch.sum(l_list_int >= l_idx)  # numbers of samples that should be calculated at level l_idx
        Ml = M0 * pow(2, l_idx)
        if l_idx == M_base:
            # in this case, nl == current_batch_size, Ml == M0 * 2^M_base
            inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), nl * Ml, replacement=True)
            inner_theta_all = theta_sample[inner_theta_idx]
            inner_theta_all[torch.linspace(0, (nl - 1) * Ml, nl.item(), dtype=torch.int32)] = sort_theta_sample
            loss_part_1 -= torch.sum(density_family.log_density_value_at_data(sort_data_sample, sort_theta_sample)
                                     + sort_theta_log_density - sort_prior_logprob)
            data_firs_expand = sort_data_sample.view(current_batch_size, 1, -1).expand(current_batch_size, Ml, -1).reshape(
                current_batch_size * Ml, -1)
            loss_part_2 += torch.sum(torch.logsumexp(
                density_family.log_density_value_at_data(data_firs_expand, inner_theta_all).view(current_batch_size, Ml) -
                prior.log_prob(inner_theta_all).view(current_batch_size, Ml), dim=1) - np.log(Ml))
        else:
            inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), nl * Ml, replacement=True)
            inner_theta_all = theta_sample[inner_theta_idx]
            inner_theta_all[torch.linspace(0, (nl - 1) * Ml, nl.item(), dtype=torch.int32)] = sort_theta_sample[:nl]
            inner_theta_all[torch.linspace(Ml - 1, nl * Ml - 1, nl.item(), dtype=torch.int32)] = sort_theta_sample[:nl]
            data_firs_expand = sort_data_sample[:nl].view(nl, 1, -1).expand(nl, Ml, -1).reshape(nl * Ml, -1)
            inner_logp = density_family.log_density_value_at_data(data_firs_expand, inner_theta_all) - prior.log_prob(inner_theta_all)
            loss_part_3 += (torch.sum(torch.logsumexp(inner_logp.view(nl, Ml), dim=1) - np.log(2)) -
                            torch.sum(torch.logsumexp(inner_logp.view(nl << 1, Ml >> 1), dim=1)) / 2) \
                           * (pow(2, mlmc_alpha * (L+1)) - 1) / (pow(2, mlmc_alpha * (L+1-l_idx)) - 1)
    return loss_part_1, loss_part_2, loss_part_3, current_batch_size


def clear_cache(c_output_density, c_output_loss, c_FileSavePath):
    if c_output_loss and c_output_density:
        dir_list = ['output_density', 'output_loss', 'output_theta', 'output_mmd', 'output_log']
        for name in dir_list:
            for root, dirs, files in os.walk(c_FileSavePath + os.sep + name):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        print("delete cache success.")


def add_vline_in_plot(x, label, color):
    value = x.item()
    plt.axvline(value, color='red')


if __name__ == '__main__':
    # parse parameter
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--mkstr', type=str, default="", help='mark_str')
    parser.add_argument('--data', type=int, default=2, help='dataset')  # 0: two_moon; 1:Lotka-Volterra; 2:M/G/1
    parser.add_argument('--ear', type=int, default=20, help='early_stop')  # 0: disable early stop; N: early stop torlarance = N;
    parser.add_argument('--mlmc', type=int, default=4, help='mlmc_type')  # 0: nested, 1: atomic, 2: RU-MLMC, 3: GRR-MLMC, 4: TGRR-MLMC
    parser.add_argument('--m0', type=int, default=8, help='mlmc_m0')  # 0: mlmc m0
    parser.add_argument('--mlmcalpha', type=float, default=1.5, help='mlmc_alpha')  # mlmc alpha value
    parser.add_argument('--clip', type=float, default=5.0, help='clip_val')
    parser.add_argument('--dbg1', type=int, default=1, help='debug_flag_1')
    parser.add_argument('--dbg2', type=int, default=4, help='debug_flag_2')
    args = parser.parse_args()
    if args.gpu == 0:
        print('using cpu')
        device = torch.device('cpu')
    else:
        print('using gpu: %d (in 1-4)' % args.gpu)
        device = torch.device("cuda:" + str(args.gpu - 1))
    mark_str = args.mkstr  # extra str for parallel running
    print("mark str: " + mark_str)
    dataset_arg = ['two_moons', 'lotka', 'mg1']
    print("using " + dataset_arg[args.data] + " dataset.")
    simulator = Dataset.Simulator(dataset_arg[args.data], device, torch.get_default_dtype())
    if args.ear == 0:
        early_stop = False
        early_stop_tolarance = 0
        print("disable early stop.")
    else:
        early_stop = True
        early_stop_tolarance = args.ear
        print("enable early stop. torlarance: %d" % early_stop_tolarance)
    mlmc_type = args.mlmc
    if mlmc_type == 0:
        print('using nested APT.')
        mlmc_func = nested_loss
    elif mlmc_type == 1:
        print('using atomic APT.')
        mlmc_func = atom_loss
    elif mlmc_type == 2:
        print('using RU-MLMC.')
        mlmc_func = ru_mlmc_loss
    elif mlmc_type == 3:
        print('using GRR-MLMC.')
        mlmc_func = grr_mlmc_loss
    elif mlmc_type == 4:
        print('using TGRR-MLMC.')
        mlmc_func = tgrr_mlmc_loss
    else:
        raise NotImplementedError
    M0 = args.m0
    mlmc_alpha = args.mlmcalpha
    mlmc_w0 = 1 - pow(2, -mlmc_alpha)
    # torch.manual_seed(327529)
    print("M0 value: %d" % M0)
    print("MLMC alpha value: %.4f" % mlmc_alpha)
    if args.clip > 1e-3:
        grad_clip = True
        grad_clip_val = args.clip
        print("using gradient clip at %.2f" % args.clip)
    else:
        grad_clip = False
    print("dbg1: %.6f, dbg2: %.6f" % (args.dbg1, args.dbg2))
    # Env setting
    plt.switch_backend("Agg")  # plt.switch_backend("tkagg")
    DefaultModelParam = SNPE_lib.DefaultModelParam()
    # put your path here
    if os.sep == "/":
        FileSavePath = "" if DefaultModelParam.linux_path is None else DefaultModelParam.linux_path
    else:
        FileSavePath = ""
    print("File Save Path: " + FileSavePath)
    plot_loss_figure_show = False
    plot_loss_figure_save = DefaultModelParam.plot_loss_figure_save  # True
    plot_mmd_figure_save = DefaultModelParam.plot_mmd_figure_save  # True
    plot_theta_figure_save = DefaultModelParam.plot_theta_figure_save  # True
    plot_density_figure_show = False  # False
    plot_density_figure_save = DefaultModelParam.plot_density_figure_save  # True
    save_theta_csv = DefaultModelParam.save_theta_csv
    clear_cuda_cache = DefaultModelParam.clear_cuda_cache
    model_save = True
    proposal_update = True
    pair_plot = False
    output_log = True
    load_trained_model = False
    model_compile = True  # only works for torch version >= 2.0 in linux
    clear_output_density = False
    clear_output_loss = clear_output_density
    debug_flag = False
    clear_cache(clear_output_density, clear_output_loss, FileSavePath)
    dim_x = simulator.dim_x
    dim_theta = simulator.dim_theta
    R = DefaultModelParam.round  # proposal update round
    N = int(DefaultModelParam.round_sample_size)  # sample generated size per round
    N_valid = int(DefaultModelParam.valid_rate * N)
    medd_samp_size = DefaultModelParam.medd_samp_size  # sample size use to evaluate median distance
    medd_round = DefaultModelParam.medd_round
    n_layer = DefaultModelParam.n_layer  # flow layers
    n_hidden = np.array([50, 50])  # hidden unit
    batch_size = DefaultModelParam.batch_size  # batch size
    steps = DefaultModelParam.steps  # steps in training network with N sample
    print_state = DefaultModelParam.print_state
    print_state_time = DefaultModelParam.print_state_time
    figure_dpi = DefaultModelParam.figure_dpi
    MMD_sample_size = DefaultModelParam.mmd_samp_size
    MMD_round = DefaultModelParam.mmd_round
    if DefaultModelParam.manual_seed is not None:
        torch.manual_seed(DefaultModelParam.manual_seed)
    ModelInfo = "Mk+APTMLMC_" + mark_str + "_Da+" + str(args.data) + "_R+" + str(R) + "_N+" + str(N) + "_Ba+" + str(batch_size) +\
                "_St+" + str(steps) + "_M0+" + str(M0) + "_alpha+" + str(int(mlmc_alpha)) + "_Ml+" + str(mlmc_type)
    if DefaultModelParam.detected_log_file:
        assert not os.path.exists(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
    if model_compile and os.sep == '/':
        # using torch.compile
        density_family_org = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device)
        density_family = torch.compile(density_family_org, mode="reduce-overhead")  # max-autotune
        print("using compiled model.")
        enable_model_compile = True
    else:
        density_family = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device)
        print("using uncompiled model.")
        enable_model_compile = False
    if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device('cuda:3'):
        # density_family.cuda(device=device)
        density_family = density_family.to(device)
        torch.cuda.empty_cache()
    # X_0
    x_0 = simulator.x_0
    # Prior
    prior = simulator.prior
    # Proposal
    proposal = prior
    geom_dist = torch.distributions.geometric.Geometric(mlmc_w0)
    # generate theta from proposal
    LossInfo = []
    LossInfo_valid = []
    LossInfo_x = 0
    LossInfo_x_list = []
    if output_log:
        output_log_idx = 0
        output_log_variables = {'round': int(),
                                'alpha': float(),
                                'mmd': float(),
                                'nlog': float(),
                                'medd': float(),
                                'c2st': float(),
                                'iter': int(),
                                'mkstr': ''}
        output_log_df = pd.DataFrame(output_log_variables, index=[])
    # define optimizer
    optimizer = torch.optim.Adam(density_family.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-4)
    # Evaluate theta
    scatter_size = 2
    plot_point = DefaultModelParam.meshgrid_plot_point  # 1000, WARN: large value will increase GPU VRAM using
    if load_trained_model:
        model_load_path = "model.pt"  # model load path
        if enable_model_compile:
            if os.sep == "\\":
                import collections
                dict_load = torch.load(model_load_path, map_location=device)
                new_state_dict = collections.OrderedDict()
                for dict_k, dict_v in dict_load.items():
                    dict_name = dict_k.replace("_orig_mod.", '')  # remove `module.`
                    new_state_dict[dict_name] = dict_v
                density_family.load_state_dict(new_state_dict)
            else:
                density_family.load_state_dict(torch.load(model_load_path, map_location=device))
        else:
            density_family.load_state_dict(torch.load(model_load_path, map_location=device))
        print("trained model load success: " + model_load_path)
    # store data for sample reuse
    full_theta = torch.tensor([], device=device)
    full_data = torch.tensor([], device=device)
    full_state_dict = []
    model_bank = []
    mmd_iter = []
    mmd_value = torch.tensor([])
    for r_idx in range(0, R):
        # theta sampling
        print("start theta sampling, round = " + str(r_idx))
        with torch.no_grad():
            proposal_sample_size = torch.tensor(N, device=device)
            if r_idx == 0 or (not proposal_update):
                # sampling theta from invariant proposal
                print("sampling theta from invariant proposal.")
                proposal_sample = proposal.sample((proposal_sample_size,))
            else:
                # sampling theta from variant proposal
                print("sampling theta from qF.")
                proposal_sample = density_family.gen_sample(proposal_sample_size, x_0)
                # resample if theta out of support
                if (not torch.all(prior.log_prob(proposal_sample) != float('-inf'))):
                    proposal_sample_in_support = proposal_sample[prior.log_prob(proposal_sample) != float('-inf')]
                    proposal_out_num = (proposal_sample_size - proposal_sample_in_support.shape[0]).item()
                    resample_times = 0
                    while True:
                        proposal_sample_extra = density_family.gen_sample(proposal_out_num * 3, x_0)
                        proposal_sample_extra_in_support = proposal_sample_extra[prior.log_prob(proposal_sample_extra) != float('-inf')]
                        proposal_sample_in_support = torch.cat((proposal_sample_in_support, proposal_sample_extra_in_support), dim=0)
                        proposal_out_num = (proposal_sample_size - proposal_sample_in_support.shape[0]).item()
                        resample_times += 1
                        if proposal_out_num <= 0:
                            proposal_sample = proposal_sample_in_support[:proposal_sample_size]
                            break
                        if resample_times == 500:
                            print('proposal sampling error!')
                            break
                    print('resample times: %d, out num: %d' % (resample_times, proposal_out_num))
                    assert torch.all(prior.log_prob(proposal_sample) != float('-inf'))
            # shuffle
            theta_sample = proposal_sample[torch.randperm(proposal_sample.shape[0])]
            # data sampling
            time_start = time.perf_counter()
            data_sample = simulator.gen_data(theta_sample)  # batch * dim_x
            time_end = time.perf_counter()
            print("%d data sampling time cost: %.2fs" % (theta_sample.shape[0], time_end-time_start))
            # sample reuse
            # print("start density calculating, round = " + str(r_idx))
            full_theta = torch.cat((full_theta, theta_sample), dim=0)
            full_data = torch.cat((full_data, data_sample), dim=0)
            perm = torch.randperm(full_theta.shape[0])
            theta_sample = full_theta[perm]  # theta_sample = full_theta
            data_sample = full_data[perm]  # data_sample = full_data
            # calculate log density
            if r_idx == 0 or (not proposal_update):
                theta_log_density = proposal.log_prob(theta_sample)
            else:
                assert theta_sample.shape[0] == (r_idx + 1) * N
                # ratio = - torch.log(torch.tensor(r_idx + 1.))
                full_log_density = proposal.log_prob(theta_sample).reshape(-1, 1)
                # model_bank = []
                prev_model = copy.deepcopy(density_family)
                model_bank.append(copy.deepcopy(density_family))
                for index in range(r_idx - 1):
                    prev_model.load_state_dict(full_state_dict[index])
                    prev_model.eval()
                    # model_bank.append(copy.deepcopy(prev_model))
                    pass
                    prev_log_density = prev_model.log_density_value_at_data(x_0.repeat([(r_idx + 1) * N, 1]), theta_sample)
                    full_log_density = torch.cat((full_log_density, prev_log_density.reshape(-1, 1)), dim=1)
                curr_log_density = density_family.log_density_value_at_data(x_0.repeat([(r_idx + 1) * N, 1]), theta_sample)
                full_log_density = torch.cat((full_log_density, curr_log_density.reshape(-1, 1)), dim=1)
                theta_log_density = torch.logsumexp(full_log_density, dim=1) - torch.log(torch.tensor(r_idx + 1.))
                full_state_dict.append(copy.deepcopy(density_family.state_dict()))
                # model_bank.append(copy.deepcopy(density_family))
            # calculate log prior
            prior_log_prob = prior.log_prob(theta_sample)
            if pair_plot:
                plot_df = pd.DataFrame(density_family.gen_sample(N, x_0).cpu())
                plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                if simulator.true_theta is not None:
                    true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                    true_theta.columns = plot_df.columns
                    g.data = true_theta
                    g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                    g.map_diag(add_vline_in_plot)
                plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' +
                            str(r_idx) + '.jpg', dpi=400)
                plt.close()
        # network training
        valid_idx = data_sample.shape[0] - N_valid
        valid_data_sample = data_sample[valid_idx:]
        valid_theta_sample = theta_sample[valid_idx:]
        valid_prior_log_prob = prior_log_prob[valid_idx:]
        valid_theta_log_density = theta_log_density[valid_idx:]
        valid_loss_best = float('inf')
        valid_loss_best_idx = 0
        training_set = torch.utils.data.TensorDataset(data_sample[:valid_idx], theta_sample[:valid_idx],
                                                      prior_log_prob[:valid_idx], theta_log_density[:valid_idx])
        dataset_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
        density_family.train()
        i = 0
        round_total_steps = steps
        while i < steps:
            # training
            for batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density in dataset_generator:
                if r_idx == 0:
                    loss = -torch.mean(density_family.log_density_value_at_data(batch_data_sample, batch_theta_sample))
                else:
                    loss_part_1, loss_part_2, loss_part_3, current_batch_size = mlmc_func(
                        batch_data_sample, batch_theta_sample, batch_prior_logprob, batch_theta_log_density)
                    loss = (loss_part_1 + loss_part_2 + loss_part_3) / current_batch_size
                LossInfo.append(loss.detach().cpu().numpy())
                LossInfo_x += (1 / len(dataset_generator))
                LossInfo_x_list.append(LossInfo_x)
                optimizer.zero_grad()  # init gradient
                loss.backward()  # calculate gradient
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(density_family.parameters(), grad_clip_val)
                optimizer.step()  # update model parameters
            # validation
            with torch.no_grad():
                if r_idx == 0:
                    valid_loss = -torch.mean(density_family.log_density_value_at_data(valid_data_sample, valid_theta_sample)).detach().cpu().numpy()
                else:
                    loss_part_1, loss_part_2, loss_part_3, current_batch_size = mlmc_func(
                        valid_data_sample, valid_theta_sample, valid_prior_log_prob, valid_theta_log_density)
                    valid_loss = ((loss_part_1 + loss_part_2 + loss_part_3) / current_batch_size).detach().cpu().numpy()
                LossInfo_valid.append(valid_loss)
                print("i: %d, valid_loss: %.4f" % (i, valid_loss))
                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    valid_loss_best_idx = i
                else:
                    if (i > (valid_loss_best_idx + early_stop_tolarance)) and early_stop:
                        round_total_steps = i + 1
                        i = steps - 1
                        print('round: %d, early stop condition satisfied.' % r_idx)
            if (i+1) % print_state_time == 0:
                # print info
                print('----------')
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print('Newest Loss: %.4f' % (LossInfo[-1]))
                print('i: %d / %d, round: %d / %d, mkstr: %s' % ((i+1), steps, r_idx, R - 1, mark_str))
            if (i+1) == steps:
                # plot loss
                density_family.eval()
                if plot_loss_figure_save:
                    plt.plot(LossInfo_x_list, LossInfo, '.', markersize=2)
                    plt.plot([loss_iter for loss_iter in range(len(LossInfo_valid))], LossInfo_valid, '.', markersize=2)
                    plt.xlabel("Number of iterations")
                    plt.ylabel("Loss")
                    plt.legend(['train loss', 'valid loss'])
                    plt.tight_layout()
                    plt.savefig(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '_' +
                                str(r_idx) + '_' + str(i+1) + '.jpg', dpi=figure_dpi)
                    pd.DataFrame(LossInfo).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss0_' + str(r_idx) + '_' + ModelInfo + '.csv')
                    pd.DataFrame(LossInfo_x_list).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss1_' + str(r_idx) + '_' + ModelInfo + '.csv')
                    pd.DataFrame(LossInfo_valid).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss2_' + str(r_idx) + '_' + ModelInfo + '.csv')
                    plt.close()
                # calculate negative log(true param)
                nlog = torch.tensor([0.])
                with torch.no_grad():
                    if simulator.true_theta is not None:
                        medd_theta_samp = density_family.gen_sample(medd_samp_size * 5, x_0)
                        medd_theta_logdensity = prior.log_prob(medd_theta_samp)
                        in_support_sample = medd_theta_samp[medd_theta_logdensity != float('-inf')]
                        normalize_ratio_estimate = in_support_sample.shape[0] / medd_theta_samp.shape[0]
                        nlog_theta = simulator.true_theta.to(device)
                        with torch.no_grad():
                            nlog = -density_family.log_density_value_at_data(x_0, nlog_theta).cpu() + np.log(normalize_ratio_estimate + 1 / 1000)
                        # print('negtative log density of true param: %.4f.' % nlog.item())
                # calculate median distance
                time_start = time.perf_counter()
                with torch.no_grad():
                    medd_result = torch.zeros(medd_round)
                    for medd_round_idx in range(medd_round):
                        medd_theta_samp = density_family.gen_sample(medd_samp_size, x_0)
                        medd_theta_samp = medd_theta_samp[prior.log_prob(medd_theta_samp) != float('-inf')]
                        if medd_theta_samp.shape[0] == 0:
                            raise ValueError
                        medd_data_samp = simulator.gen_data(medd_theta_samp)
                        medd_result[medd_round_idx] = torch.nanmedian(torch.norm((medd_data_samp - x_0) / simulator.scale, dim=1)).cpu()
                    medd = torch.nanmedian(medd_result)
                time_end = time.perf_counter()
                time_medd = time_end - time_start
                # print('median distance: %.4f.' % medd)
                # calculate c2st
                time_start = time.perf_counter()
                if DefaultModelParam.calc_c2st:
                    c2st = metrics.c2st(simulator.reference_theta.cpu(), medd_theta_samp.cpu())
                else:
                    c2st = torch.tensor([0.])
                time_end = time.perf_counter()
                time_c2st = time_end - time_start
                # calculate mmd
                time_start = time.perf_counter()
                mmd = metrics.mmd(simulator.reference_theta, medd_theta_samp)
                time_end = time.perf_counter()
                time_mmd = time_end - time_start
                print('medd: %.4f, time: %.2fs, c2st: %.4f, time: %.2fs, mmd: %.4f, time: %.2fs, nlog: %.4f, mkstr: %s' %
                      (medd.item(), time_medd, c2st.item(), time_c2st, mmd.item(), time_mmd, nlog.item(), mark_str))
                # save qF theta sample as csv file
                # if save_theta_csv and r_idx == (R-1):
                if save_theta_csv:
                    pd.DataFrame(medd_theta_samp.cpu()).to_csv(FileSavePath + 'output_theta' + os.sep + ModelInfo + '_' +
                                        str(r_idx) + '_' + str(i+1) + '.csv')
                    if pair_plot:
                        plot_df = pd.DataFrame(medd_theta_samp.cpu())
                        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                        if simulator.true_theta is not None:
                            true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                            true_theta.columns = plot_df.columns
                            g.data = true_theta
                            g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                            g.map_diag(add_vline_in_plot)
                        plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' +
                                    str(r_idx) + '_' + str(i + 1) + '.jpg', dpi=400)
                        plt.close()
                # clear cache
                if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device('cuda:3'):
                    with torch.cuda.device(device):
                        if clear_cuda_cache:
                            torch.cuda.empty_cache()
                density_family.train()
            i += 1
        # proposal update
        if output_log:
            output_log_df.loc[len(output_log_df.index)] = [r_idx + 1, mlmc_alpha, mmd.item(), nlog.item(),
                                                           medd.item(), c2st.item(), round_total_steps, mark_str]
    # Evaluate model
    density_family.eval()
    with torch.no_grad():
        # save loss csv file
        pd.DataFrame(LossInfo).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '.csv')
        if plot_loss_figure_show:  # plot loss
            plt.figure()
            plt.plot(range(0, len(LossInfo), 1), LossInfo, '.', markersize=2)
            plt.show()
        if output_log:  # save log file
            output_log_df.to_csv(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
        if model_save:
            # save model
            torch.save(density_family.state_dict(), FileSavePath + 'output_model' + os.sep + ModelInfo + ".pt")

