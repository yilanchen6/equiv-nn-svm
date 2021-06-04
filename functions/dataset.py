import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MyMnist(Dataset):
    def __init__(self, train=True, size=None, normalize=True):
        if normalize:
            mean = (0.1307,)
            std = (0.3081,)
        else:
            mean = (0.,)
            std = (1.,)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        self.mnist = torchvision.datasets.MNIST(root='./data', train=train,
                                                download=True, transform=transform)
        subset_indices = ((self.mnist.targets == 0) + (self.mnist.targets == 1)).nonzero(as_tuple=False).view(-1)
        if size:
            subset_indices = subset_indices[:size]
        self.mnist.data = self.mnist.train_data[subset_indices]
        self.mnist.targets = self.mnist.targets[subset_indices]
        self.mnist.targets[self.mnist.targets == 0] = -1

    def __getitem__(self, index):
        data, target = self.mnist[index]

        return index, data, target

    def __len__(self):
        return len(self.mnist)
