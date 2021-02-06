import torch
import pandas
import numpy as np
import scipy.sparse as sp


def get_hd5():
    pass

# 数据需要处理成的形式 [batch, time_step, num_nodes, in_features]
# ---------------- [  N  ,    C     ,    H     ,     W      ]


x = np.array(
    [[[[1.,2.,3.,4.], [5,6,7,8], [9,10,11,12]],
     [[13,14,15,16], [17,18,19,20],[21,22,23,24]]],
     [[[1., 2., 3., 4.], [5, 6, 7, 8], [9, 10, 11, 12]],
      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
     ]
)
x = np.array(
    [[[1.,2.,3.,4.], [5,6,7,8], [9,10,11,12]],
     [[13,14,15,16], [17,18,19,20],[21,22,23,24]]]
)
y = torch.from_numpy(x).to(torch.float)
# print(y.shape)
# print(y.permute(0,2,1).shape)

# vec1 = torch.from_numpy(np.array([1, 2, 3, 4]))
# vec2 = torch.from_numpy(np.array([5, 6, 7, 8]))
# print(vec1 * vec2)
# mat1 = torch.ones((5, 2))
# index = indices = torch.tensor([[4, 2, 1], [2, 0, 2]])
# values = torch.tensor([3, 4, 5], dtype=torch.float32)
# mat2 = torch.sparse_coo_tensor(indices=indices, values=values, size=[5, 5])

# mat2 = torch.ones((4, 10, 2, 3))
# print(mat2)
# print(torch.matmul(mat2, mat1))
# print(torch.einsum("ij,kljm->", mat1, mat2))
# torch.sparse.einsum()
# conv = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3))
# z = conv(y)
# print(z.shape)
