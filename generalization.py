import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

from models.model import FCNN
from functions.dataset import MyMnist
from functions.hparams import HParam
from functions.sub_functions import check_dir
from train_sgd import test_nn

import neural_tangents as nt
from neural_tangents import stax


def cal_kernel(fcnn, X, optimizer):
    # \phi(x) gradient
    phi_x = []
    for i, x in enumerate(X):
        optimizer.zero_grad()
        output = fcnn(x)
        output.backward()
        grads = []
        for param in fcnn.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        # print(grads.shape)
        phi_x.append(grads.detach())
    phi_x = torch.stack(phi_x)

    tangent_kernel = torch.matmul(phi_x, phi_x.T)

    return tangent_kernel



def train(config, device, save_path):
    width = config.model.width
    hidden_layer = config.model.hidden_layer
    epochs = config.train_net.epochs
    lr = config.train_net.lr
    lam = config.train_net.lam
    size = config.data.size
    print(width, hidden_layer, epochs, lr, lam, size)
    save_path += str(width) + '_' + str(hidden_layer) + '_' + str(epochs) + '_' + str(lr) \
                 + '_' + str(lam) + '_' + str(size) + '/'
    check_dir(save_path)


    # # FCNN
    # _, _, kernel_fn = stax.serial(
    #     stax.Dense(512), stax.Relu(),
    #     # stax.Dense(512), stax.Relu(),
    #     stax.Dense(1)
    # )
    # # calculate NTK
    # kernel_fn_batched = nt.utils.batch.batch(kernel_fn, device_count=-1, batch_size=size)
    # X_numpy = torch.flatten(X.cpu(), 1).numpy()
    # ntk = kernel_fn_batched(X_numpy, X_numpy, 'ntk')
    # ntk = np.array(ntk)
    # np.save('/home/yilan/svm_submit/ntk/gene_128.npy', ntk)

    # FCNN
    fcnn = FCNN(width=width, hidden_layer=hidden_layer).to(device)
    optimizer = torch.optim.SGD(fcnn.parameters(), lr)

    # test data
    test_data = MyMnist(train=False, normalize=config.data.normalize)
    N_test = len(test_data)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    init_test = torch.zeros(N_test, device=device)
    with torch.no_grad():
        for i, (ids, X_batch, Y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            init_test[ids] = fcnn(X_batch).flatten()

    acc_test_nn = []
    loss_test_nn = []

    # train data
    train_data = MyMnist(train=True, size=size, normalize=config.data.normalize)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    _, X, Y = next(iter(train_loader))
    X = X.to(device=device)
    Y = Y.to(device=device)
    N = X.size(0)
    init_out = fcnn(X).detach()
    loss_nn = []
    output_nn = []


    # kernel_integral_rect = 0.
    # a_integral_rect = 0.
    # hinge_loss_grad = 0.
    b = 0.
    dt = lr
    output_km = []
    loss_km = []


    o_integral = 0
    output = 0
    o_int = 0
    output_km2 = []

    delta = 0.99
    constant = 3 * torch.sqrt(torch.log(torch.tensor(2 / delta)) / 2 / N)
    print('constant in bound: ', constant)
    loss_bound = []


    for epoch in tqdm(range(epochs)):

        # test whole test set
        acc_test, loss_test = test_nn(fcnn, test_loader, init_test)
        acc_test_nn.append(acc_test)
        loss_test_nn.append(loss_test)

        # tangent kernel
        tg = cal_kernel(fcnn, X, optimizer)


        # train NN and get the hinge_loss_grad for this step
        optimizer.zero_grad()
        output = fcnn(X) - init_out
        output = output.flatten()

        loss, hinge, hinge_loss_grad = soft_margin_loss(output.flatten(), Y, fcnn.classifier[-1].weight / (width ** 0.5), lam)
        loss.backward()
        optimizer.step()

        output_nn.append(output.flatten().detach().cpu().numpy())
        loss_nn.append(hinge.item())


        # # calculate the kernel machine
        t = epoch * dt
        # kernel_step = torch.abs(hinge_loss_grad) * tg * np.exp(lam * t) * dt
        kernel_step = hinge_loss_grad * tg * np.exp(lam * t) * dt

        if t == 0:
            kernel0 = kernel_step
            kernel_integral_trape = torch.zeros_like(tg)
            kernel_integral_rect = torch.zeros_like(tg)
        else:
            kernel_integral_trape = kernel_integral_rect + (kernel0 + kernel_step) / 2  # Trapezoidal rule
            kernel_integral_rect += kernel_step        # Rectangle rule


        K = np.exp(-lam * t) * kernel_integral_trape
        # if t == 0:
        #     a = torch.zeros_like(hinge_loss_grad).float()
        # else:
        #     # a = - a_integral_trape / kernel_integral_trape / size
        #     a = - torch.sign(hinge_loss_grad) / N

        a = - torch.ones_like(hinge_loss_grad) / size

        # o_km0 = torch.sum(a * K, dim=-1)
        o_km = torch.matmul(K, a)
        if torch.isnan(o_km).sum() > 0:
            ValueError('There is nan in kernel output!')
        hinge, _ = hinge_loss(o_km, Y)
        output_km.append(o_km.flatten().detach().cpu().numpy())
        loss_km.append(hinge.item())


        # generalization bound
        max_B = torch.sqrt(a @ K @ a)
        scale = torch.sqrt(torch.trace(K)) / size
        # scale0 = torch.sum(torch.sqrt(torch.diag(K))) / size
        radmacher = max_B * scale
        gap = 2 * radmacher + constant

        bound = hinge + gap
        loss_bound.append(bound.item())


        # another method by integrating the equation directly
        if t == 0:
            o_km2 = torch.zeros_like(o_km)
        else:
            o_integral += lam * output * dt
            o_int += - hinge_loss_grad / size * tg * dt
            o_km2 = torch.sum(o_int, dim=-1) - o_integral
        output_km2.append(o_km2.flatten().detach().cpu().numpy())


    output_nn = np.array(output_nn)
    loss_nn = np.array(loss_nn)
    loss_test_nn = np.array(loss_test_nn)
    np.savez(save_path + '/output_nn.npz', output_nn=output_nn, loss_nn=loss_nn, loss_test_nn=loss_test_nn)


    output_km = np.array(output_km)
    output_km2 = np.array(output_km2)
    loss_km = np.array(loss_km)
    loss_bound = np.array(loss_bound)
    np.savez(save_path + '/output_km.npz', output_nn=output_km, output_km2=output_km2, loss_km=loss_km, loss_bound=loss_bound)


    # plot
    loss_path = save_path + '/output.png'
    step = np.arange(epochs) * lr
    plt.figure(figsize=(18, 5), dpi=500)

    ax = plt.subplot(1, 3, 1)
    # ax = plt.gca()
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(step, output_nn[:, i], linestyle='--', color=color, label='NN')
        plt.plot(step, output_km[:, i], linestyle='-', color=color, label='km')
        if i == 0:
            plt.legend()
    ax.set_xscale("log", base=10)
    plt.xlabel('step', fontsize=18)
    plt.title('Outputs', fontsize=18)

    ax = plt.subplot(1, 3, 2)
    diff = output_nn - output_km
    diff2 = output_nn - output_km2
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(step, diff[:, i], linestyle='-', color=color, label='NN - km')
        # plt.plot(step, diff2[:, i], linestyle='--', color=color, label='NN - km2')
        if i == 0:
            plt.legend()
    ax.set_xscale("log", base=10)
    plt.xlabel('step', fontsize=18)
    plt.title('Difference of outputs', fontsize=18)

    ax = plt.subplot(1, 3, 3)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step, loss_nn, linestyle='-', color=color, label='NN train')
    plt.plot(step, loss_km, linestyle='--', color=color, label='km train')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step, loss_test_nn, linestyle='dotted', color=color, label='NN test')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(step, loss_bound, linestyle='None', marker='.', color=color, label='true loss bound')
    ax.set_xscale("log", base=10)
    plt.legend()
    plt.xlabel('step', fontsize=18)
    plt.title('Hinge loss', fontsize=18)
    plt.tight_layout()

    plt.savefig(loss_path)
    plt.show()
    plt.close()


def hinge_loss(output, target):
    # L = (1 - target * inputs).clamp(min=0)

    hinge_loss = 1 - torch.mul(output, target)
    hinge_loss[hinge_loss < 0] = 0

    hinge_loss_grad = - target
    hinge_loss_grad[hinge_loss < 0] = 0.
    # hinge_loss_grad = - target[hinge_loss > 0] / output.size(0)

    return torch.mean(hinge_loss), hinge_loss_grad


def soft_margin_loss(output, target, parameter, lam=0.01):
    hinge, hinge_loss_grad = hinge_loss(output, target)
    loss = hinge + 0.5 * lam * torch.norm(parameter) ** 2
    return loss, hinge, hinge_loss_grad


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change GPU here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/svm_gene.yaml")
    args, unknown = parser.parse_known_args()
    config = HParam(args.config)

    today = datetime.date.today().isoformat()
    save_path = 'exper/' + config.data.dataset + '_gene/' + str(today) + '/'

    train(config, device, save_path)
