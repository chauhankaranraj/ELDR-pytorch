import pdb
import numpy as np
from sklearn.datasets import load_iris, load_boston
import sklearn.preprocessing

import torch
from torch.utils.data import Dataset, DataLoader


class IrisDataset(Dataset):
    """
    Helper Dataset class to load Iris dataset in a pytorch training pipeline
    """
    def __init__(self, transform=None, target_transform=None):
        super(IrisDataset, self).__init__()
        self.X, self.y = load_iris(return_X_y=True)

        self.y = self.y.reshape(-1, 1)
        # # scale data
        # self.scaler = getattr(sklearn.preprocessing, scaler)()
        # self.X = self.scaler.fit_transform(self.X)
        
        # any other transforms for data
        self.transform = transform
        self.target_transform = target_transform

        # apply transform once, not every time item is fetched
        # NOTE: this is feasible when we have a small dataset
        # otherwise it's better to do it in getitem
        if self.transform is not None:
            self.X = self.transform(self.X)
        if self.target_transform is not None:
            self.y = self.target_transform(self.y)
        
    def __getitem__(self, index):
        """Returns item at index, as well as index itself

        Args:
            index (int): index into iris dataset

        Returns:
            (torch.Tensor), (torch.Tensor): data and corresponding indices
        """
        # values to return
        # NOTE: must convert to float type otherwise pytorch internals complain
        x_ret = torch.from_numpy(self.X[index]).type(torch.float)
        # y_ret = torch.from_numpy(self.y[index]).type(torch.float)
        index_ret = torch.from_numpy(np.array(self.y[index]).reshape(1,)).type(torch.float)

        # if self.transform is not None:
        #     x_ret = self.transform(x_ret)
        # if self.target_transform is not None:
        #     index

        return x_ret, index_ret

    def __len__(self):
        return len(self.X)


class BostonHousingDataset(Dataset):
    """
    Helper Dataset class to load BostonHousing dataset in a pytorch training pipeline
    """
    def __init__(self, transform=None, target_transform=None):
        super(BostonHousingDataset, self).__init__()
        self.X, self.y = load_boston(return_X_y=True)

        # for batch indexing
        self.y = self.y.reshape(-1, 1)
        
        # any other transforms for data
        self.transform = transform
        self.target_transform = target_transform

        # apply transform once, not every time item is fetched
        # NOTE: this is feasible when we have a small dataset
        # otherwise it's better to do it in getitem
        if self.transform is not None:
            self.X = self.transform(self.X)
        if self.target_transform is not None:
            self.y = self.target_transform(self.y)
        
    def __getitem__(self, index):
        """Returns data point item at index, as well as its value

        Args:
            index (int): index into boston dataset

        Returns:
            (torch.Tensor), (torch.Tensor): data and corresponding labels
        """
        # values to return
        # NOTE: must convert to float type otherwise pytorch internals complain
        x_ret = torch.from_numpy(self.X[index]).type(torch.float)
        y_ret = torch.from_numpy(self.y[index]).type(torch.float)
        return x_ret, y_ret

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    dataset = BostonHousingDataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch_data, batch_labels in train_loader:
        print(batch_data.shape)