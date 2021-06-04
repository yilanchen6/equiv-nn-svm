import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

import neural_tangents as nt
from neural_tangents import stax

from models.model import FCNN
from functions.dataset import MyMnist
from functions.hparams import HParam
from functions.sub_functions import check_dir, set_deterministic
from functions.plot_utils import plot_sgd


def cal_ntk(hidden_layer, X_train, X_test, save_name):
    # FCNN
    layers = [stax.Dense(512), stax.Relu()]
    for _ in range(hidden_layer):
        layers += [stax.Dense(512), stax.Relu()]
    layers += [stax.Dense(1)]
    _, _, kernel_fn = stax.serial(*layers)

    # calculate NTK
    # kernel_fn = jit(kernel_fn, static_argnums=(2,))
    kernel_fn_batched = nt.utils.batch.batch(kernel_fn, device_count=-1, batch_size=17)
    X_train = torch.flatten(X_train, 1).numpy()
    X_test = torch.flatten(X_test, 1).numpy()
    ntk_train = kernel_fn_batched(X_train, X_train, 'ntk')
    ntk_train_test = kernel_fn_batched(X_train, X_test, 'ntk')
    ntk_train = np.array(ntk_train)
    ntk_train_test = np.array(ntk_train_test)
    check_dir('./ntk/')
    np.savez(save_name, ntk_train=ntk_train, ntk_train_test=ntk_train_test)
    print('NTK calculation done!')

    return ntk_train, ntk_train_test


def train_svm(config, ntk_train, ntk_train_test, train_data, Y_train, Y_test, sub_ids_train, sub_ids_test, epochs,
              step_test, lr, lam, batch_size, device, save_path, **kwargs):
    # SVM / infinite NN
    ntk_train = torch.from_numpy(ntk_train).to(device)
    ntk_train_test = torch.from_numpy(ntk_train_test).to(device)
    N_train = ntk_train.shape[0]
    alpha = torch.zeros(N_train, device=device)
    ntk_sub_train = ntk_train[sub_ids_train]
    ntk_sub_test = ntk_train_test[:, sub_ids_test]
    Y_train, Y_test = Y_train.to(device), Y_test.to(device)

    loss_train_svm = []
    loss_test_svm = []
    acc_train_svm = []
    acc_test_svm = []
    output_train_svm = []
    output_test_svm = []
    set_deterministic(config.model.seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    t = 1
    for epoch in tqdm(range(epochs)):
        for i, (ids, X, Y) in enumerate(train_loader):
            if t in step_test:
                # test acc and test loss
                g_test = torch.matmul(alpha, ntk_train_test)
                hinge_test = hinge_loss(g_test, Y_test)
                acc_test = cal_acc(g_test, Y_test)
                loss_test_svm.append(hinge_test.item())
                acc_test_svm.append(acc_test.item())

                # test output
                g_sub_test = torch.matmul(alpha, ntk_sub_test)
                output_test_svm.append(g_sub_test.flatten().detach().cpu().numpy())

                # train acc and train loss
                g_train = torch.matmul(ntk_train, alpha)
                acc_train = cal_acc(g_train, Y_train)
                acc_train_svm.append(acc_train.item())
                hinge = hinge_loss(g_train, Y_train)
                loss_train_svm.append(hinge.item())

                # train output
                g_sub_train = torch.matmul(ntk_sub_train, alpha)
                output_train_svm.append(g_sub_train.flatten().detach().cpu().numpy())

            # train sgd
            ntk_batch = ntk_train[ids]
            Y = Y.to(device)
            g_x = torch.matmul(ntk_batch, alpha)
            alpha[ids] = (1 - lr * lam) * alpha[ids] + (lr / ids.size(0)) * (Y * g_x < 1) * Y
            t += 1

    loss_train_svm = np.array(loss_train_svm)
    loss_test_svm = np.array(loss_test_svm)
    acc_train_svm = np.array(acc_train_svm)
    acc_test_svm = np.array(acc_test_svm)
    output_train_svm = np.array(output_train_svm)
    output_test_svm = np.array(output_test_svm)
    svm_save = dict(loss_train_svm=loss_train_svm, loss_test_svm=loss_test_svm, acc_train_svm=acc_train_svm,
                    acc_test_svm=acc_test_svm, output_train_svm=output_train_svm, output_test_svm=output_test_svm)
    np.savez(save_path + '/output_svm.npz', **svm_save)
    if config.train_net.save:
        torch.save(alpha, save_path + '/alpha.pt')

    return svm_save


def train_nn(config, train_data, test_data, X_sub_train, X_sub_test, sub_ids_train, sub_ids_test,
             width, hidden_layer, batch_size, epochs, lr, lam, step_test, device, save_path, **kwargs):
    N_train, N_test = len(train_data), len(test_data)

    # FCNN
    fcnn = FCNN(width=width, hidden_layer=hidden_layer, seed=config.model.seed).to(device)

    # f_0(X)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    init_train = torch.zeros(N_train, device=device)
    with torch.no_grad():
        for i, (ids, X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            init_train[ids] = fcnn(X).flatten()
    init_sub_train = init_train[sub_ids_train].to(device)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    init_test = torch.zeros(N_test, device=device)
    with torch.no_grad():
        for i, (ids, X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            init_test[ids] = fcnn(X).flatten()
    init_sub_test = init_test[sub_ids_test].to(device)

    loss_train_nn = []
    loss_test_nn = []
    acc_train_nn = []
    acc_test_nn = []
    output_train_nn = []
    output_test_nn = []
    optimizer = torch.optim.SGD(fcnn.parameters(), lr)
    set_deterministic(config.model.seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    t = 1
    for epoch in tqdm(range(epochs)):
        for i, (ids, X, Y) in enumerate(train_loader):
            if t in step_test:
                # test output
                with torch.no_grad():
                    f_sub_test = fcnn(X_sub_test).flatten() - init_sub_test
                    output_test_nn.append(f_sub_test.cpu().numpy())

                acc_test, loss_test = test_nn(fcnn, test_loader, init_test)
                acc_test_nn.append(acc_test)
                loss_test_nn.append(loss_test)

                # train output
                with torch.no_grad():
                    f_sub_train = fcnn(X_sub_train).flatten() - init_sub_train
                    output_train_nn.append(f_sub_train.detach().cpu().numpy())

                # train acc and train loss
                acc_train, loss_train = test_nn(fcnn, train_loader, init_train)
                acc_train_nn.append(acc_train)
                loss_train_nn.append(loss_train)

            # train sgd
            X, Y = X.to(device), Y.to(device)
            init_batch = init_train[ids].to(device)
            optimizer.zero_grad()
            output = fcnn(X).flatten() - init_batch
            loss, hinge = soft_margin_loss(output, Y, fcnn.classifier[-1].weight / (width ** 0.5), lam)
            loss.backward()
            optimizer.step()
            t += 1

    loss_train_nn = np.array(loss_train_nn)
    loss_test_nn = np.array(loss_test_nn)
    acc_train_nn = np.array(acc_train_nn)
    acc_test_nn = np.array(acc_test_nn)
    output_train_nn = np.array(output_train_nn)
    output_test_nn = np.array(output_test_nn)
    nn_save = dict(loss_train_nn=loss_train_nn, loss_test_nn=loss_test_nn, acc_train_nn=acc_train_nn,
                   acc_test_nn=acc_test_nn, output_train_nn=output_train_nn, output_test_nn=output_test_nn)
    np.savez(save_path + '/output_nn.npz', **nn_save)
    if config.train_net.save:
        torch.save(fcnn.state_dict(), save_path + '/fcnn.pt')

    return nn_save


@torch.no_grad()
def test_nn(fcnn, test_loader, init_test):
    # test
    correct = torch.tensor(0., device=device)
    hinge_test = torch.tensor(0., device=device)
    with torch.no_grad():
        for i, (ids_batch, X_batch, Y_batch) in enumerate(test_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            init_batch = init_test[ids_batch].to(device)
            output = fcnn(X_batch).flatten() - init_batch
            # round the numerical error in pytorch
            output[torch.abs(output) < 1e-6] = 0.
            correct += torch.sum((torch.sign(output) == Y_batch).float())
            hinge = 1 - torch.mul(output, Y_batch)
            hinge[hinge < 0] = 0
            hinge_test += torch.sum(hinge)

    N_test = len(test_loader.dataset)
    acc_test = correct.item() / N_test
    loss_test = hinge_test.item() / N_test
    return acc_test, loss_test


def hinge_loss(output, target):
    # L = (1 - target * inputs).clamp(min=0)

    hinge_loss = 1 - torch.mul(output, target)
    hinge_loss[hinge_loss < 0] = 0
    return torch.mean(hinge_loss)


def soft_margin_loss(output, target, parameter, lam=0.001):
    hinge = hinge_loss(output, target)
    loss = hinge + 0.5 * lam * torch.norm(parameter) ** 2
    return loss, hinge


def cal_acc(output, target):
    # round the numerical error in pytorch
    output[torch.abs(output) < 1e-6] = 0.
    acc = torch.mean((torch.sign(output) == target).float())
    return acc


def train(config, device, save_path):
    seed = config.model.seed
    width = config.model.width
    hidden_layer = config.model.hidden_layer
    epochs = config.train_net.epochs
    lr = config.train_net.lr
    lam = config.train_net.lam
    batch_size = config.train_net.batch_size
    print(width, hidden_layer, epochs, lr, lam, batch_size, seed)
    save_path += str(width) + '_' + str(hidden_layer) + '_' + str(epochs) + '_' + str(lr) \
                 + '_' + str(lam) + '_' + str(batch_size) + '_' + str(seed) + '/'
    check_dir(save_path)
    train_para = dict(seed=config.model.seed,
                      width=config.model.width,
                      hidden_layer=config.model.hidden_layer,
                      epochs=config.train_net.epochs,
                      lr=config.train_net.lr,
                      lam=config.train_net.lam,
                      batch_size=config.train_net.batch_size,
                      normalize=config.data.normalize,
                      device=device,
                      save_path=save_path)
    print(train_para)

    train_data = MyMnist(train=True, normalize=config.data.normalize)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    test_data = MyMnist(train=False, normalize=config.data.normalize)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    _, X_train, Y_train = next(iter(train_loader))
    _, X_test, Y_test = next(iter(test_loader))

    # random select a subset samples to draw
    N_train, N_test = X_train.size(0), X_test.size(0)
    np.random.seed(config.model.seed)
    sub_ids_train = np.random.choice(range(N_train), size=5)
    sub_ids_test = np.random.choice(range(N_test), size=5)
    X_sub_train = X_train[sub_ids_train].to(device)
    X_sub_test = X_test[sub_ids_test].to(device)
    train_para['sub_ids_train'] = sub_ids_train
    train_para['sub_ids_test'] = sub_ids_test

    ntk_path = './ntk/mnist_all_' + str(hidden_layer) + ('' if config.data.normalize else '_un-normalize') + '.npz'
    if os.path.exists(ntk_path):
        ntk_train = np.load(ntk_path)['ntk_train']
        ntk_train_test = np.load(ntk_path)['ntk_train_test']
    else:
        print('No ntk. Will calculate it.')
        ntk_train, ntk_train_test = cal_ntk(hidden_layer, X_train, X_test, ntk_path)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # step to test acc
    num_test = epochs * config.train_net.test_num_every_epoch
    total_step = epochs * len(train_loader)
    step_test = np.round(np.logspace(0, np.log10(total_step), num_test)).astype(int)
    train_para['step_test'] = step_test

    svm_save = None
    if config.train_net.train_svm:
        # train SVM
        svm_save = train_svm(config, ntk_train, ntk_train_test, train_data, Y_train, Y_test, **train_para)

    nn_save = None
    if config.train_net.train_nn:
        # trian NN
        nn_save = train_nn(config, train_data, test_data, X_sub_train, X_sub_test, **train_para)

    if svm_save and nn_save:
        # plot
        loss_path = save_path + '/plot_sgd.png'
        plot_sgd(loss_path, step_test, **svm_save, **nn_save)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # change GPU here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/svm_sgd.yaml")
    args, unknown = parser.parse_known_args()
    config = HParam(args.config)

    today = datetime.date.today().isoformat()
    save_path = 'exper/' + config.data.dataset + '_sgd' + ('' if config.data.normalize else '_un-normalize') \
                + '/' + str(today) + '/'

    train(config, device, save_path)
