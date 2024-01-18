import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import my_lib
from scipy.special import softmax
from matplotlib import pyplot as plt
from scipy.io import savemat
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system setting
num_wd = 180     # number of Wireless devices
num_ap = 10      # number of base stations (access points)
num_time_slots = 400   # total number of time slots
dataset_len = 50   # size of the dataset for DNN-2
batch_size = 50     # size of each training batch


bandwidth_uplink = 100 * np.random.rand(num_ap).astype(np.float32)
bandwidth_uplink[0] = 45
bandwidth_fronthaul = 1000 * np.random.rand(num_ap).astype(np.float32)

delta_uplink = 0.15 + 0.35 * np.random.rand(num_wd, num_ap).astype(np.float32)
delta_fronthaul = 0.1 + 0 * np.random.rand(num_wd, num_ap).astype(np.float32)

dataset = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
learning_rate = 0.01
model = my_lib.DNN1(num_wd, num_ap).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.3)

dataset1 = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
model1 = my_lib.DNN1(num_wd, num_ap).to(device)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.3)

cost_alg_ot = []
cost_gcg_ot = []
cost_bsl_ot = []
time_gcg_ot = []
cost_opt_ot = []
time_opt_ot = []
loss_alg_ot = []

for epoch in range(num_time_slots):
    # beginning of each time slot
    sizes_data = 3 + 7 * np.random.rand(num_wd).astype(np.float32)  # current system states
    delta_uplink = delta_uplink * (1 + 0.01 * np.random.randn(num_wd, num_ap))
    delta_uplink = np.maximum(np.minimum(delta_uplink, 0.5), 0.15).astype(np.float32)

    sys_state_i = np.zeros(num_wd + num_ap*num_wd).astype(np.float32)
    sys_state_i[:num_wd] = sizes_data
    sys_state_i[num_wd:] = delta_uplink.reshape(-1)

    # gcg solver
    decision_gcg, cost_gcg, _ = my_lib.gcg_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
    # optimal solver
    decision_opt, cost_opt = my_lib.gurobi_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
    # initialize decision of our alg
    decision_alg = torch.zeros(num_wd, num_ap)

    # inference bsl
    decision_hat_bsl = model1(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
    decision_bsl_real = fun.softmax(decision_hat_bsl, dim=1)
    decision_bsl_real = decision_bsl_real.cpu().detach().numpy()
    num_candidates = 20
    for num_cdt in range(num_candidates):
        decision_bsl = np.zeros([num_wd, num_ap])
        for i in range(num_wd):
            idx_i = np.random.choice(num_ap, p=decision_bsl_real[i, :])
            decision_bsl[i, idx_i] = 1
        if num_cdt == 0:
            cost_bsl_opt = my_lib.obj_val(decision_bsl, delta_uplink, bandwidth_uplink, sizes_data) + my_lib.obj_val(decision_bsl, delta_fronthaul, bandwidth_fronthaul, sizes_data)
            decision_bsl_opt = copy.deepcopy(decision_bsl)
        else:
            cost_bsl_current = my_lib.obj_val(decision_bsl, delta_uplink, bandwidth_uplink, sizes_data) + my_lib.obj_val(decision_bsl, delta_fronthaul, bandwidth_fronthaul, sizes_data)
            if cost_bsl_current < cost_bsl_opt:
                cost_bsl_opt = copy.deepcopy(cost_bsl_current)
                decision_bsl_opt = copy.deepcopy(decision_bsl)

    if epoch < len(dataset):
        if epoch == 0:
            # inference
            decision_hat_real = model(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
            index = torch.argmax(decision_hat_real, dim=1)
            index = index.cpu().detach().numpy()  # offloading decisions of each user

            # get sample
            x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)
            x_item1, y_item1 = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_bsl_opt).reshape(-1)

            loss = my_lib.cross_entropy_loss(decision_hat_real, y_item.to(device))
            loss1 = my_lib.cross_entropy_loss(decision_hat_bsl, y_item1.to(device))
            print(f'{loss.item():.6f}, {loss1.item():.6f}')
            # print(decision_hat_real.shape, y_item.shape)
            # print(decision_hat_real.dim(), y_item.dim())
            # update dataset
            dataset.replace_item(x_item, y_item)

        elif epoch < len(dataset):
            # inference
            decision_hat_real = model(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
            index = torch.argmax(decision_hat_real, dim=1)
            index = index.cpu().detach().numpy()  # offloading decisions of each user

            # learning process
            lst1 = list(range(epoch))
            train_set = torch.utils.data.Subset(dataset, lst1)
            dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model(d_in_i)
            loss = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # get sample
            x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)
            # update dataset
            dataset.replace_item(x_item, y_item)

            # learning process
            lst1 = list(range(epoch))
            train_set = torch.utils.data.Subset(dataset1, lst1)
            dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model1(d_in_i)
            loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            # get sample
            x_item1, y_item1 = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_bsl_opt).reshape(-1)
            # update dataset
            dataset1.replace_item(x_item1, y_item1)
    else:
        # inference
        decision_hat_real = model(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
        index = torch.argmax(decision_hat_real, dim=1)
        index = index.cpu().detach().numpy()  # offloading decisions of each user

        # learning process
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        d_in_i, d_out_i = dataiter.next()
        d_in_i = d_in_i.to(device)
        d_out_i = d_out_i.to(device)
        outputs = model(d_in_i)
        loss = my_lib.cross_entropy_loss(outputs, d_out_i)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get sample
        x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)
        # update dataset
        dataset.replace_item(x_item, y_item)

        # learning process DROO
        dataloader = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        d_in_i, d_out_i = dataiter.next()
        d_in_i = d_in_i.to(device)
        d_out_i = d_out_i.to(device)
        outputs = model1(d_in_i)
        loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        # get sample
        x_item1, y_item1 = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_bsl_opt).reshape(-1)
        # update dataset
        dataset1.replace_item(x_item1, y_item1)

    # calculate objetive value (computing latency of our alg)
    y_real = decision_hat_real.cpu().detach().numpy()
    for i in range(num_wd):
        decision_alg[i, index[i]] = 1
    cost_alg = my_lib.obj_val(decision_alg, delta_uplink, bandwidth_uplink, sizes_data) + my_lib.obj_val(decision_alg, delta_fronthaul, bandwidth_fronthaul, sizes_data)
    cost_alg_ot.append(cost_alg)
    cost_bsl_ot.append(cost_bsl_opt)
    cost_gcg_ot.append(cost_gcg)
    cost_opt_ot.append(cost_opt)
    loss_alg_ot.append(loss.item())

    if epoch % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_time_slots}], Loss: {loss.item():.6f}, cost: {cost_alg.item()/cost_opt.item():.6f}, cost-bsl: {cost_bsl_opt.item()/cost_opt.item():.6f}')

# plot and plot performance
ratio = np.array(cost_alg_ot).reshape(-1)/np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg
ratio1 = np.array(cost_bsl_ot).reshape(-1)/np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg

# calculating moving average competitive ratio
move_ave_size = 20  # moving average subset size
ratio_mean = np.zeros(len(ratio) - move_ave_size)
ratio1_mean = np.zeros(len(ratio) - move_ave_size)
ratio_min = np.zeros(len(ratio) - move_ave_size)
ratio_max = np.zeros(len(ratio) - move_ave_size)
for i in range(len(ratio)-move_ave_size):
    ratio_mean[i] = ratio[i:i+move_ave_size].mean()
    ratio1_mean[i] = ratio1[i:i + move_ave_size].mean()
    ratio_max[i] = ratio[i:i+move_ave_size].max()
    ratio_min[i] = ratio[i:i+move_ave_size].min()

ratio_cvg = ratio[-100:].mean()

# plot the average approximation ratio
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.axhline(y=ratio_cvg, color='peru', linestyle='-', label="approximation\n ratio")
l1, = ax1.plot(list(range(num_time_slots-move_ave_size)), ratio_mean, 'k-', label="moving\naverage")
l3, = ax1.plot(list(range(num_time_slots-move_ave_size)), ratio1_mean, color='darkslategrey', linestyle='-')
l2 = ax1.fill_between(list(range(num_time_slots-move_ave_size)), ratio_min, ratio_max, color='silver', label='range')
ax1.set_ylabel('Normalized Communication Latency', fontsize=15)
ax1.set_xlabel('Time Slots', fontsize=12)
ax1.set_xlim([-1, len(ratio)-move_ave_size])
ax1.set_ylim([1, 2.0])
ax1.grid(axis='both')
l4, = ax2.plot(list(range(num_time_slots-move_ave_size)), loss_alg_ot[:len(ratio_mean)], 'r-', label="Loss1")
ax2.set_ylabel('Loss 1', fontsize=15)
ax2.set_ylim([0, 3])
plt.legend([l1, l2, l3, l4], ["Moving average cost of DRL-TOBS", "Range",  "Moving average cost of DROO [11]", "Loss of DNN-1"])
# plt.savefig("../plots/figure7.pdf", format="pdf", bbox_inches='tight')
plt.show()
