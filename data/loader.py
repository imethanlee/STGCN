import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power


class Utils:
    mean = 0
    std = 0

    @staticmethod
    def fit(x: np.ndarray):
        len_x = len(x)
        Utils.mean = np.sum(x) / len_x
        Utils.std = np.std(x)

    @staticmethod
    def z_score(x: np.ndarray):
        z = (x - Utils.mean) / Utils.std
        return z

    @staticmethod
    def inverse_z_score(z: np.ndarray):
        x = z * Utils.std + Utils.mean
        return x

    @staticmethod  # 存档，可能在其他项目用
    def weight_matrix_preprocessing(file_path, normalized_k=0.1):
        dist_mat = pd.read_csv(file_path, header=None).to_numpy()
        std = np.std(dist_mat.flatten())
        adj_mat = np.exp(-np.square(dist_mat / std))
        adj_mat[adj_mat < normalized_k] = 0
        return adj_mat


class TrafficFlowData:
    def __init__(self, device: torch.device, v_path: str, w_path: str, len_train: int, len_val: int,
                 in_timesteps: int = 12, out_timesteps: int = 3):
        self.device = device
        self.v_path = v_path
        self.w_path = w_path
        self.len_train = len_train
        self.len_val = len_val
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.train = None
        self.val = None
        self.test = None
        self.w_adj_mat = self.get_weighted_adjacency_matrix()
        self.num_nodes = self.w_adj_mat.shape[1]
        self.gen_data()

    def get_weighted_adjacency_matrix(self, sigma2=0.1, epsilon=0.5, scaling=True):
        df = pd.read_csv(self.w_path, header=None).to_numpy()
        return df

    def get_data(self):
        df = pd.read_csv(self.v_path, header=None)
        train_data = df[: self.len_train].to_numpy()
        val_data = df[self.len_train: self.len_train + self.len_val].to_numpy()
        test_data = df[self.len_train + self.len_val:].to_numpy()
        return train_data, val_data, test_data

    def transform_data(self, data: np.ndarray):
        # transform from row data to formatted data
        len_record = len(data)
        num_available_data = len_record - self.in_timesteps - self.out_timesteps

        x = np.zeros([num_available_data, self.in_timesteps, self.num_nodes, 1])
        y = np.zeros([num_available_data, self.num_nodes])

        for i in range(num_available_data):
            start = i
            end = i + self.in_timesteps
            x[i, :, :, :] = data[start: end].reshape(self.in_timesteps, self.num_nodes, 1)
            y[i] = data[end + self.out_timesteps - 1]

        return x, y

    def gen_data(self):
        # generate formatted data
        train_data, val_data, test_data = self.get_data()
        train_data_x, train_data_y = self.transform_data(train_data)
        val_data_x, val_data_y = self.transform_data(val_data)
        test_data_x, test_data_y = self.transform_data(test_data)
        Utils.fit(np.hstack((train_data_x.flatten(), train_data_y.flatten())))
        train_data_x = torch.Tensor(Utils.z_score(train_data_x)).to(self.device)
        train_data_y = torch.Tensor(Utils.z_score(train_data_y)).to(self.device)
        val_data_x = torch.Tensor(Utils.z_score(val_data_x)).to(self.device)
        val_data_y = torch.Tensor(Utils.z_score(val_data_y)).to(self.device)
        test_data_x = torch.Tensor(Utils.z_score(test_data_x)).to(self.device)
        test_data_y = torch.Tensor(Utils.z_score(test_data_y)).to(self.device)
        # test_data_y = torch.Tensor(test_data_y).to(self.device)

        self.train = TensorDataset(train_data_x, train_data_y)
        self.val = TensorDataset(val_data_x, val_data_y)
        self.test = TensorDataset(test_data_x, test_data_y)

    def get_conv_kernel(self, approx: str):
        if approx == "Linear":
            W_wave = np.eye(self.num_nodes, self.num_nodes) + self.w_adj_mat
            D_wave = np.diag(np.sum(W_wave, axis=1))
            kernel = np.matmul(
                np.matmul(fractional_matrix_power(D_wave, -0.5), W_wave),
                fractional_matrix_power(D_wave, -0.5)
            )
            # return torch.Tensor(kernel).to(self.device)
            # convert to sparse tensor
            r, c, v = [], [], []
            for i in range(len(kernel)):
                for j in range(len(kernel[0])):
                    if kernel[i, j] != 0:
                        r.append(i)
                        c.append(j)
                        v.append(kernel[i, j])
            index = torch.LongTensor([r, c])
            value = torch.FloatTensor(v)
            sparse_kernel = torch.sparse.FloatTensor(index, value, [self.num_nodes, self.num_nodes])
            return sparse_kernel.to(self.device)
        elif approx == "Cheb":
            pass
        else:
            raise ValueError("No such type!")

