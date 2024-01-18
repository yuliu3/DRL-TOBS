import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import my_lib
import math
import random
from matplotlib import pyplot as plt
from scipy.io import savemat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system setting
num_wd = 100     # number of Wireless devices
num_sv = 10      # number of Edge servers
num_time_slots = 420   # total number of time slots
dataset_len = 100   # size of the dataset for DNN-2
batch_size = 50     # size of each training batch

capacities_sv = np.array([3.584, 3.712, 6.656, 6.144, 6.400, 7.168, 7.424, 3.072, 6.656, 3.200]).reshape(num_sv, 1).astype(np.float32)
delta = 0.5 + 0.5*np.random.rand(num_wd, num_sv).astype(np.float32)


dataset0 = my_lib.MyDataset(set_length=dataset_len, num_users=num_wd, num_devices=num_sv)
dataset1 = my_lib.MyDataset(set_length=dataset_len, num_users=num_wd, num_devices=num_sv)
dataset2 = my_lib.MyDataset(set_length=dataset_len, num_users=num_wd, num_devices=num_sv)

learning_rate = 0.01
model0 = my_lib.DNN2(num_wd, num_sv).to(device)  # model using proposed approximation algorithm
optimizer0 = torch.optim.SGD(model0.parameters(), lr=learning_rate, momentum=0.9)
model1 = my_lib.DNN2(num_wd, num_sv).to(device)  # model using commercial gurobi solver
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)

cost_alg0_ot = []
cost_alg1_ot = []

cost_gcg_ot = []
time_gcg_ot = []
cost_opt_ot = []
time_opt_ot = []

loss_alg0_ot = []
loss_alg1_ot = []

# the devices that will be shut down
wd_shutdown_lst = random.sample(range(num_wd), 10)
indices = []
for idx in range(num_wd):
    if idx not in wd_shutdown_lst:
        indices.append(idx)
wd_shutdown = np.ones(num_wd).astype(np.float32)
wd_shutdown[wd_shutdown_lst] = np.zeros(len(wd_shutdown_lst)).astype(np.float32)
t_down = 150
t_start = 300

for epoch in range(num_time_slots):
    # beginning of each time slot
    task_sizes = 0.6 + 1.9 * np.random.rand(num_wd).astype(np.float32)  # current system states
    if epoch == t_down:
        sd = model0.state_dict()
        model1.load_state_dict(sd)
    if epoch in range(t_down, t_start):
        task_sizes = task_sizes * wd_shutdown
    # gcg solver
    decision_gcg, cost_gcg, _ = my_lib.gcg_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
    # optimal solver
    decision_opt, cost_opt = my_lib.gurobi_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
    # initialize decision of our alg
    decision_alg0 = torch.zeros(num_wd, num_sv)
    decision_alg1 = torch.zeros(num_wd, num_sv)

    # if epoch in range(t_down, t_start) and epoch % 10 == 0:
    #     print(decision_gcg[wd_shutdown_lst, :].argmax(axis=1))

    # inference
    decision_hat_0 = model0(torch.from_numpy(task_sizes).reshape(1, num_wd).to(device)).reshape(num_wd, num_sv)
    index0 = torch.argmax(decision_hat_0, dim=1)
    index0 = index0.cpu().detach().numpy()  # offloading decisions of each user

    decision_hat_1 = model1(torch.from_numpy(task_sizes).reshape(1, num_wd).to(device)).reshape(num_wd, num_sv)
    index1 = torch.argmax(decision_hat_1, dim=1)
    index1 = index1.cpu().detach().numpy()  # offloading decisions of each user

    # get sample
    x_item, y_item = torch.from_numpy(task_sizes).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

    # learning process model0
    if epoch in range(t_down, t_start):
        epoch1 = epoch - t_down
        if epoch == t_down:
            loss0 = my_lib.cross_entropy_loss(decision_hat_0, y_item.to(device))
        else:
            if epoch1 < len(dataset1):
                lst1 = list(range(epoch1))
                train_set = torch.utils.data.Subset(dataset1, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch1 if epoch1 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss_down(outputs, d_out_i, torch.IntTensor(indices).to(device))
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()
    elif epoch < t_down:
        if epoch == 0:
            loss0 = my_lib.cross_entropy_loss(decision_hat_0, y_item.to(device))
        else:
            if epoch < len(dataset0):
                lst1 = list(range(epoch))
                train_set = torch.utils.data.Subset(dataset0, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
            # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()
    elif epoch >= t_start:
        epoch2 = epoch - t_start
        if epoch2 == 0:
            loss0 = my_lib.cross_entropy_loss(decision_hat_0, y_item.to(device))
        else:
            if epoch2 < len(dataset2):
                lst2 = list(range(epoch2))
                train_set = torch.utils.data.Subset(dataset2, lst2)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch2 if epoch2 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()

    # leaning model1
    if epoch in range(t_down, t_start):
        epoch1 = epoch - t_down
        if epoch == t_down:
            loss1 = my_lib.cross_entropy_loss(decision_hat_1, y_item.to(device))
        else:
            if epoch1 < len(dataset1):
                lst1 = list(range(epoch1))
                train_set = torch.utils.data.Subset(dataset1, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch1 if epoch1 < batch_size else batch_size,
                                        shuffle=True)
            else:
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
    elif epoch < t_down:
        if epoch == 0:
            loss1 = my_lib.cross_entropy_loss(decision_hat_1, y_item.to(device))
        else:
            if epoch < len(dataset0):
                lst1 = list(range(epoch))
                train_set = torch.utils.data.Subset(dataset0, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
            # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model1(d_in_i)
            loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
    elif epoch >= t_start:
        epoch2 = epoch - t_start
        if epoch2 == 0:
            loss1 = my_lib.cross_entropy_loss(decision_hat_1, y_item.to(device))
        else:
            if epoch2 < len(dataset2):
                lst2 = list(range(epoch2))
                train_set = torch.utils.data.Subset(dataset2, lst2)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch2 if epoch2 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model1(d_in_i)
            loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

    # update dataset
    if epoch in range(t_down, t_start):
        dataset1.replace_item(x_item, y_item)
    elif epoch >= t_start:
        dataset2.replace_item(x_item, y_item)
    else:
        dataset0.replace_item(x_item, y_item)

    # calculate objetive value (computing latency of our alg)
    # y_real = decision_hat_real.cpu().detach().numpy()
    for i in range(num_wd):
        decision_alg0[i, index0[i]] = 1
        decision_alg1[i, index1[i]] = 1
    cost_alg0 = my_lib.obj_val(decision_alg0, delta, capacities_sv, task_sizes)
    cost_alg1 = my_lib.obj_val(decision_alg1, delta, capacities_sv, task_sizes)
    cost_alg0_ot.append(cost_alg0)
    cost_alg1_ot.append(cost_alg1)
    cost_gcg_ot.append(cost_gcg)
    cost_opt_ot.append(cost_opt)
    loss_alg0_ot.append(loss0.item())
    loss_alg1_ot.append(loss1.item())

    if epoch % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_time_slots}], Loss0: {loss0.item():.6f}, Loss1: {loss1.item():.6f}, cost0: {cost_alg0.item() / cost_opt.item():.6f}, cost1: {cost_alg1.item() / cost_opt.item():.6f}')


# plot and plot performance
ratio0 = np.array(cost_alg0_ot).reshape(-1) / np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg
ratio1 = np.array(cost_alg1_ot).reshape(-1) / np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg
ratio1[:t_down] = ratio0[:t_down]
loss_alg1_ot[:t_down] = loss_alg0_ot[:t_down]

# calculating moving average competitive ratio
move_ave_size = 5  # moving average subset size
ratio0_mean = np.zeros(len(ratio0) - move_ave_size)
ratio0_min = np.zeros(len(ratio0) - move_ave_size)
ratio0_max = np.zeros(len(ratio0) - move_ave_size)
ratio1_mean = np.zeros(len(ratio1) - move_ave_size)
ratio1_min = np.zeros(len(ratio1) - move_ave_size)
ratio1_max = np.zeros(len(ratio1) - move_ave_size)
for i in range(len(ratio0) - move_ave_size):
    ratio0_mean[i] = ratio0[i:i + move_ave_size].mean()
    ratio0_max[i] = ratio0[i:i + move_ave_size].max()
    ratio0_min[i] = ratio0[i:i + move_ave_size].min()

    ratio1_mean[i] = ratio1[i:i + move_ave_size].mean()
    ratio1_max[i] = ratio1[i:i + move_ave_size].max()
    ratio1_min[i] = ratio1[i:i + move_ave_size].min()

ratio_cvg = ratio0[-100:].mean()

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# l1, = ax1.plot(list(range(len(ratio0_mean))), ratio0_mean[:len(ratio0_mean)], 'k-', label="moving\naverage")
# # l2 = ax1.fill_between(list(range(num_time_slots-move_ave_size)), ratio_min, ratio_max, color='silver', label='range')
# ax1.set_ylabel('Normalized Processing Latency', fontsize=15)
# ax1.set_xlabel('Time Slots', fontsize=12)
# ax1.set_xlim([-1, len(ratio0)])
# l3, = ax2.plot(list(range(len(ratio0_mean))), loss_alg0_ot[:len(ratio0_mean)], 'r-', label="Loss1")
# ax2.set_ylabel('Loss 2', fontsize=15)
# ax2.set_ylim([0, math.ceil(max(loss_alg0_ot))])
# plt.legend([l1, l3], ["Normalized Latency", "Loss_2, active wds"])
# plt.grid()
# # plt.savefig("./plots/fig4.pdf", format="pdf", bbox_inches='tight')
# plt.show()
#
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# l1, = ax1.plot(list(range(len(ratio1_mean))), ratio1_mean[:len(ratio1_mean)], 'k-')
# # l2 = ax1.fill_between(list(range(num_time_slots-move_ave_size)), ratio_min, ratio_max, color='silver', label='range')
# ax1.set_ylabel('Normalized Processing Latency', fontsize=15)
# ax1.set_xlabel('Time Slots', fontsize=12)
# ax1.set_xlim([-1, len(ratio1)])
# l3, = ax2.plot(list(range(len(ratio1_mean))), loss_alg1_ot[:len(ratio1_mean)], 'r-')
# ax2.set_ylabel('Loss 2', fontsize=15)
# ax2.set_ylim([0, math.ceil(max(loss_alg1_ot))])
# plt.legend([l1, l3], ["Normalized Latency", "Loss_2, all wds"])
# plt.grid()
# # plt.savefig("./plots/fig4.pdf", format="pdf", bbox_inches='tight')
# plt.show()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
l1, = ax1.plot(list(range(len(ratio0_mean))), ratio0_mean[:len(ratio0_mean)], color="black", linestyle='-')
l2, = ax1.plot(list(range(len(ratio1_mean))), ratio1_mean[:len(ratio1_mean)], color="grey", linestyle='--')
ax1.set_ylabel('Normalized Processing Latency', fontsize=15)
ax1.set_xlabel('Time Slots', fontsize=12)
ax1.set_xlim([-1, len(ratio0_mean)])
ax1.set_ylim([1, 2])
l3, = ax2.plot(list(range(len(ratio0_mean))), loss_alg0_ot[:len(ratio0_mean)], color="firebrick", linestyle='-')
l4, = ax2.plot(list(range(len(ratio0_mean))), loss_alg1_ot[:len(ratio0_mean)], color="tomato", linestyle='--')
ax2.set_ylim([-1, 3])
ax2.set_ylabel("Loss of DNN-2", fontsize=15)
plt.legend([l1, l2, l3, l4], ["Normalized Latency, using Loss of active WDs", "Normalized Latency, using Loss of all WDs",
                              "Loss of DNN-2, using Loss of active WDs", "Loss of DNN-2, using Loss of all WDs"], loc=1)
plt.grid()
plt.savefig("./plots/fig4.pdf", format="pdf", bbox_inches='tight')
plt.show()