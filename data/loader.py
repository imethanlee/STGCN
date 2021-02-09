import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power


class Utils:
    mean = 0
    std = 0

    @staticmethod
    def z_score(x: np.ndarray):
        x_flatten = x.flatten()
        len_x = len(x_flatten)
        Utils.mean = np.sum(x_flatten) / len_x
        Utils.std = np.std(x_flatten)
        z = (x - Utils.mean) / Utils.std
        return z


"""以下代码针对已经处理好的PeMS(228)数据集"""


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

    def get_weighted_adjacency_matrix(self):
        df = pd.read_csv(self.w_path, header=None)
        return df.to_numpy()

    def get_data(self):
        df = pd.read_csv(self.v_path, header=None)
        train_data = Utils.z_score(df[: self.len_train].to_numpy())
        val_data = Utils.z_score(df[self.len_train: self.len_train + self.len_val].to_numpy())
        test_data = Utils.z_score(df[self.len_train + self.len_val:].to_numpy())
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
            y[i] = data[end + self.out_timesteps - 1] * Utils.std + Utils.mean

        return torch.Tensor(x).to(self.device), torch.Tensor(y).to(self.device)

    def gen_data(self):
        # generate formatted data
        train_data, val_data, test_data = self.get_data()
        train_data_x, train_data_y = self.transform_data(train_data)
        val_data_x, val_data_y = self.transform_data(val_data)
        test_data_x, test_data_y = self.transform_data(test_data)
        self.train = TensorDataset(train_data_x, train_data_y)
        self.val = TensorDataset(val_data_x, val_data_y)
        self.test = TensorDataset(test_data_x, test_data_y)

    def get_conv_kernel(self, approx: str):
        if approx == "Linear":
            A_wave = np.eye(self.num_nodes, self.num_nodes) + self.w_adj_mat
            degree_arr = A_wave.diagonal()
            degree_mat = np.diag(degree_arr)
            kernel = np.matmul(
                np.matmul(fractional_matrix_power(degree_mat, -0.5), A_wave),
                fractional_matrix_power(degree_mat, -0.5))
            return torch.Tensor(kernel).to(self.device)
        elif approx == "Cheb":
            pass
        else:
            raise ValueError("No such type!")


# tfd = TrafficFlowData("PeMS_V_228.csv", "PeMS_W_228.csv", 200, 200)
# print(tfd.train_data_x.shape, tfd.train_data_y.shape)
# print(tfd.val_data_x.shape, tfd.val_data_y.shape)
# print(tfd.test_data_x.shape, tfd.test_data_y.shape)


"""以下代码针对METR数据集"""
