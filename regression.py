import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from functions.dataset import MyMnist
from functions.sub_functions import check_dir


hidden_layer = 0
normalize = False
save_path = 'exper/regression/'
check_dir(save_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change GPU here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_data = MyMnist(train=True, normalize=normalize)
train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
test_data = MyMnist(train=False, normalize=normalize)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
_, X_train, Y_train = next(iter(train_loader))
_, X_test, Y_test = next(iter(test_loader))


ntk_path = './ntk/mnist_all_' + str(hidden_layer) + ('' if normalize else '_un-normalize') + '.npz'
ntk_train = np.load(ntk_path)['ntk_train']
ntk_train_test = np.load(ntk_path)['ntk_train_test']

ntk_train = torch.from_numpy(ntk_train).to(device)
ntk_train_test = torch.from_numpy(ntk_train_test).to(device)
N_train = ntk_train.shape[0]
Y_train, Y_test = Y_train.to(device).float(), Y_test.to(device).float()


# NTK ridge regression
lam_list = [0.] + [10.0 ** i for i in range(-3, 4)]
for lam in lam_list:
    H_inverse = torch.inverse(ntk_train + lam * 0.5 * torch.eye(N_train, device=device))
    alpha = torch.matmul(H_inverse, Y_train)
    torch.save(alpha, save_path + '/alpha_lam_' + str(lam) + '.pt')
    predict_train = torch.matmul(ntk_train, alpha)
    acc_train = torch.mean((torch.sign(predict_train) == Y_train).float())

    predict_test = torch.matmul(alpha, ntk_train_test)
    acc_test = torch.mean((torch.sign(predict_test) == Y_test).float())
    print(lam, acc_train, acc_test)


