import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim


class TemporalConv(nn.Module):
    """
        :param x: [batch, time_step, num_nodes, in_features].
        :param kt: int, kernel size of temporal convolution.
        :param in_channel: int, size of input channel.
        :param out_channel: int, size of output channel.
        :param activation: str, activation function.
        :return: tensor, [batch_size, time_step-Kt+1, in_features, num_nodes].
    """
    def __init__(self, kt: int, in_channel: int, out_channel: int, activation: str = "GLU"):
        super(TemporalConv, self).__init__()
        self.kt = kt
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.activation = activation
        if self.activation == "GLU":
            self.conv = nn.Conv2d(in_channels=in_channel,
                                  out_channels=2 * out_channel,
                                  kernel_size=(1, kt))
        else:
            self.conv = nn.Conv2d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=(1, kt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute: [batch,time_step,num_nodes,in_feature]
        #       -> [batch,in_feature,num_nodes,time_step]
        x_permute = x.permute(0, 3, 2, 1)
        x_conv = self.conv(x_permute)
        if self.activation == "GLU":
            p = x_conv[:, :self.out_channel, :, :]
            q = x_conv[:, -self.out_channel:, :, :]
            r = p + x_permute
            tc_out = r * F.sigmoid(q)
        else:
            raise ValueError("No such activation")
        tc_out = tc_out.permute(0, 3, 2, 1)
        return tc_out


class GraphConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, approx="Linear", use_bias=True):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.approx = approx
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channel))
        self.init_params()

    def init_params(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.xavier_uniform_(self.bias)

    def forward(self, kernel_mat: torch.Tensor, input_mat: torch.Tensor) -> torch.Tensor:
        output_mat = None
        # Graph Convolution Approximation: 1. Chebshev 2. Linear
        if self.approx == "Cheb":
            output_mat = None
        elif self.approx == "Linear":
            mat = torch.matmul(input_mat, self.weight)
            output_mat = torch.matmul(kernel_mat, mat)
        # add bias
        if self.use_bias:
            output_mat += self.bias

        return output_mat


class STConvBlock(nn.Module):
    def __init__(self):
        super(STConvBlock, self).__init__()
        self.tc_block1 = TemporalConv()
        self.gc_block = GraphConv()
        self.tc_block2 = TemporalConv()

    def forward(self, x, kernel) -> torch.Tensor:
        output1 = self.tc_block1(x)
        output2 = F.relu(self.gc_block(kernel, output1))
        output3 = self.tc_block2(output2)
        return output3


class Model(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_features: int,
                 num_timesteps_input: int,
                 num_timesteps_output: int,
                 kernel_mat: torch.Tensor):
        super(Model, self).__init__()
        self.block1 = STConvBlock()
        self.block2 = STConvBlock()
        self.fully_conn = nn.Linear(in_features=0,
                                    out_features=num_timesteps_output,
                                    bias=True)

    def forward(self, x) -> torch.Tensor:
        output_block1 = self.block1(x)
        output_block2 = self.block2(output_block1)
        output = self.linear(output_block2)
        return output
