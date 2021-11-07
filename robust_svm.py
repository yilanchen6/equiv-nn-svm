import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from functions.dataset import MyMnist
from ibp import network_bounds
from models.model import FCNN
from collections import OrderedDict


def h_map(u):
    return 2 * u * (np.pi - torch.arccos(u)) + torch.sqrt(1 - u ** 2)


def binary_search(m, l, r, condition):
    if condition > 0:
        l = m
    else:
        r = m
    m = (l + r) / 2
    return m, l, r


def bound_ntk(delta, d, x_norm, x_X_train_dot, X_train_norm):
    h_minimum = - 0.429738
    # x1 = - 0.394233486
    h_x_min = - 0.794
    sqrt_d = torch.sqrt(d)
    pi = torch.tensor(np.pi, device=x_norm.device)

    x_norm_L = x_norm - sqrt_d * delta
    if x_norm_L < 0:
        print('x_norm_min less than 0')
        x_norm_L[x_norm_L < 0.] = 0.
    x_norm_U = x_norm + sqrt_d * delta

    dot_min = x_X_train_dot - sqrt_d * delta * X_train_norm
    u_min = torch.zeros_like(dot_min)
    u_min[dot_min >= 0] = dot_min[dot_min >= 0] / x_norm_U
    u_min[dot_min < 0] = dot_min[dot_min < 0] / x_norm_L
    u_min = u_min / X_train_norm

    dot_max = x_X_train_dot + sqrt_d * delta * X_train_norm
    u_max = torch.zeros_like(dot_max)
    u_max[dot_max >= 0] = dot_max[dot_max >= 0] / x_norm_L
    u_max[dot_max < 0] = dot_max[dot_max < 0] / x_norm_U
    u_max = u_max / X_train_norm
    u_max[u_max >= 1.] = 1.

    h = torch.zeros((2, u_max.shape[0]), device=x_norm.device)
    h_min, h_max = h_map(u_min), h_map(u_max)
    h[0] = torch.minimum(h_min, h_max)
    h[1] = torch.maximum(h_min, h_max)
    index = torch.logical_and(u_min <= h_x_min, u_max >= h_x_min).nonzero(as_tuple=False).view(-1)
    if index.size()[0] != 0:
        h[0, index] = h_minimum
    Sigma = torch.zeros_like(h)
    Sigma[0][h[0] >= 0] = x_norm_L * h[0][h[0] >= 0]
    Sigma[0][h[0] < 0] = x_norm_U * h[0][h[0] < 0]
    Sigma[1][h[1] >= 0] = x_norm_U * h[1][h[1] >= 0]
    Sigma[1][h[1] < 0] = x_norm_L * h[1][h[1] < 0]
    Sigma = Sigma * X_train_norm / (2 * pi * d)

    return Sigma


def test_svm(alpha, test_classified_right, X_train, ids, X_test, Y_test, ntk_train_test, T, device):
    g_test = torch.matmul(alpha, ntk_train_test)

    d = torch.tensor(X_test.size()[-1], device=device)
    X_train_norm = torch.norm(X_train, dim=-1).to(device)
    pi = torch.tensor(np.pi, device=device)
    delta_list = []

    for i, (x, y) in enumerate(zip(X_test, Y_test)):
        x_norm = torch.norm(x)
        x_X_train_dot = torch.matmul(X_train, x)
        u = x_X_train_dot / (x_norm * X_train_norm)
        Sigma0 = x_norm * X_train_norm * h_map(u) / (2 * pi * d)
        g0 = torch.matmul(alpha, Sigma0)
        g0_sign = torch.sign(g0)
        if i == 0:  # check if ntk value is consistent with that calculated by library
            if torch.mean(ntk_train_test[:, ids[0]] - Sigma0) > 1e-6 or abs(g0 - g_test[ids[0]]) > 1e-4:
                ValueError('NTK not correct!')

        if test_classified_right and g0_sign != y:
            print('Classified wrong!')
            continue
        delta = 0.05
        delta_l = 0.
        delta_r = 0.3
        temp_list = []

        for i in range(T):

            Sigma = bound_ntk(delta, d, x_norm, x_X_train_dot, X_train_norm)
            if g0_sign > 0:
                g = torch.matmul(alpha[alpha > 0], Sigma[0][alpha > 0]) + torch.matmul(alpha[alpha < 0],
                                                                                       Sigma[1][alpha < 0])
            elif g0_sign < 0:
                g = torch.matmul(alpha[alpha > 0], Sigma[1][alpha > 0]) + torch.matmul(alpha[alpha < 0],
                                                                                       Sigma[0][alpha < 0])
            else:
                ValueError('g0_sign error!')

            if g0_sign * g >= 0:
                temp_list.append(delta)

                if g0_sign * g < 1e-3:
                    break

            delta, delta_l, delta_r = binary_search(delta, delta_l, delta_r, g0_sign * g)

            if i == T - 1:
                print('Not Find!')
        delta_list.append(np.max(temp_list))

    # print(delta_list)
    print(len(delta_list), np.mean(delta_list), np.std(delta_list))

    return np.mean(delta_list)


def test_nn(width, nn_paths, test_classified_right, X_test, Y_test, T, device):
    nn_delta_list = []
    for j, nn_path in enumerate(nn_paths):
        print(nn_path)
        acc_test_nn = np.load(nn_path + '/output_nn.npz')['acc_test_nn']
        print(j + 1, 'acc_test_nn: ', acc_test_nn[-1])
        seed = nn_path[-1]
        fcnn_init = FCNN(width=width, hidden_layer=0, seed=int(seed)).to(device)

        nn_path += '/fcnn.pt'
        state_dict = torch.load(nn_path)
        if type(state_dict) is OrderedDict:
            fcnn = FCNN(width=width, hidden_layer=0).to(device)
            fcnn.load_state_dict(state_dict)
        else:
            fcnn = state_dict.to(device)

        fcnn.eval()
        delta_list = []

        with torch.no_grad():
            # for x in X_test:
            for i, (x, y) in enumerate(zip(X_test, Y_test)):
                x = torch.unsqueeze(x, 0)
                init_out = fcnn_init(x)
                f0 = fcnn(x) - init_out
                f0_sign = torch.sign(f0)
                if test_classified_right and f0_sign != y:
                    print('Classified wrong!')
                    continue

                delta = 0.01
                delta_l = 0.
                delta_r = 0.3
                temp_list = []

                for t in range(T):
                    upper, lower = network_bounds(fcnn.classifier, x, delta)
                    upper, lower = upper - init_out, lower - init_out

                    if f0_sign > 0:
                        f_bound = lower
                    elif f0_sign < 0:
                        f_bound = upper
                    else:
                        ValueError('f0_sign error!')

                    if f0_sign * f_bound >= 0:
                        temp_list.append(delta)

                        if f0_sign * f_bound < 1e-3:
                            break

                    delta, delta_l, delta_r = binary_search(delta, delta_l, delta_r, f0_sign * f_bound)

                    if t == T - 1:
                        print('Not Find!')
                delta_list.append(np.max(temp_list))

        # print(len(delta_list), delta_list)
        print(len(delta_list), np.mean(delta_list), np.std(delta_list))
        nn_delta_list.append(np.mean(delta_list))

    print(nn_delta_list)
    print(np.mean(nn_delta_list), np.std(nn_delta_list))


def test(model):
    model = str(model)
    width = int(1e4)
    seed = 0
    normalize = False
    T = 1000
    print(model, width)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change GPU here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nn_path_normalize = {'10000': ['your paths to NN trained by normalized data', ], }
    # nn_path_un_normalize = {'10000': ['your paths to NN trained by un-normalized data',],}
    nn_path_un_normalize = {'10000': ['/home/yilan/svm_submit/exper/mnist_sgd_un-normalize/2021-06-04/10000_0_100_1_0.001_64_6', ], }

    # paths to NN weights
    if normalize:
        nn_paths = nn_path_normalize[str(width)]
    else:
        nn_paths = nn_path_un_normalize[str(width)]

    # path to svm kernel weights
    if normalize:
        svm_paths = ['your paths to svm kernel weights',]
    else:
        svm_paths = ['your paths to svm kernel weights', ]

    train_data = MyMnist(train=True, normalize=normalize)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    _, X_train, Y_train = next(iter(train_loader))
    X_train = torch.flatten(X_train, 1).to(device)

    for (data_size, test_classified_right) in [(100, True), ('all', False), ('all', True)]:
        print('\n', data_size, test_classified_right)

        test_data = MyMnist(train=False, normalize=normalize)
        if data_size == 'all':
            test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        else:
            test_loader = DataLoader(test_data, batch_size=data_size, shuffle=True)
        torch.manual_seed(seed)
        ids, X_test, Y_test = next(iter(test_loader))
        X_test = torch.flatten(X_test, 1).to(device)
        Y_test = Y_test.to(device)

        if model == 'svm':
            # load ntk
            ntk_path = './ntk/mnist_all_0' + ('' if normalize else '_un-normalize') + '.npz'
            ntk_train_test = np.load(ntk_path)['ntk_train_test']
            ntk_train_test = torch.from_numpy(ntk_train_test).to(device)

            svm_delta_list = []
            for j, svm_path in enumerate(svm_paths):
                print('\n', svm_path)
                acc_test_svm = np.load(svm_path + '/output_svm.npz')['acc_test_svm']
                print(j + 1, 'acc_test_nn: ', acc_test_svm[-1])
                svm_path += '/alpha.pt'
                alpha = torch.load(svm_path).to(device)
                delta_mean = test_svm(alpha, test_classified_right, X_train, ids, X_test, Y_test, ntk_train_test, T,
                                      device)

                svm_delta_list.append(delta_mean)

            print(svm_delta_list)
            print(np.mean(svm_delta_list), np.std(svm_delta_list))

        elif model == 'nn':
            test_nn(width, nn_paths, test_classified_right, X_test, Y_test, T, device)


def test_regressions():
    seed = 0
    data_size = 'all'
    path = 'exper/regression/'

    test_classified_right = False
    normalize = False
    T = 1000

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change GPU here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = MyMnist(train=True, normalize=normalize)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    _, X_train, Y_train = next(iter(train_loader))
    X_train = torch.flatten(X_train, 1).to(device)

    test_data = MyMnist(train=False, normalize=normalize)
    if data_size == 'all':
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    else:
        test_loader = DataLoader(test_data, batch_size=data_size, shuffle=True)
    torch.manual_seed(seed)
    ids, X_test, Y_test = next(iter(test_loader))
    X_test = torch.flatten(X_test, 1).to(device)
    Y_test = Y_test.to(device)

    # load ntk
    ntk_path = './ntk/mnist_all_0' + ('' if normalize else '_un-normalize') + '.npz'
    ntk_train_test = np.load(ntk_path)['ntk_train_test']
    ntk_train_test = torch.from_numpy(ntk_train_test).to(device)

    alpha_list = os.listdir(path)
    for f in alpha_list:
        print(f)
        alpha = torch.load(path + "/" + f).to(device)
        delta_mean = test_svm(alpha, test_classified_right, X_train, ids, X_test, Y_test, ntk_train_test, T, device)


if __name__ == '__main__':
    test('nn')
    # test_regressions()
