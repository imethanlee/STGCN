import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


class TemporalChannelAlign(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(TemporalChannelAlign, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if self.in_channels > self.out_channels:
            # reduce channels by convolution
            x_tca = self.conv(x)
        elif self.in_channels < self.out_channels:
            # add channels by padding 0
            batch_size, channels, num_nodes, timesteps = x.shape
            x_tca = torch.cat([x, torch.zeros(batch_size, self.out_channels - channels, num_nodes, timesteps).to(x)], dim=1)
        else:
            x_tca = x
        return x_tca


class TemporalConv(nn.Module):
    """
        :param x: [batch, time_step, num_nodes, in_features].
        :param kt: int, kernel size of temporal convolution.
        :param in_channel: int, size of input channel.
        :param out_channel: int, size of output channel.
        :param activation: str, activation function.
        :return: tensor, [batch_size, time_step-Kt+1, in_features, num_nodes].
    """
    def __init__(self, kt: int, in_channels: int, out_channels: int, activation: str = "GLU"):
        super(TemporalConv, self).__init__()
        self.kt = kt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        if self.activation == "GLU":
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=2 * out_channels,
                                  kernel_size=(1, kt))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kt))
        self.tca = TemporalChannelAlign(in_channels=in_channels,
                                        out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute: [batch,time_step,num_nodes,in_channels]
        # -------> [batch,in_channels,num_nodes,time_step]
        x_permute = x.permute(0, 3, 2, 1)
        x_conv = self.conv(x_permute)
        x_tca = self.tca(x_permute)[:, :, :, self.kt - 1:]
        if self.activation == "GLU":
            p = x_conv[:, :self.out_channels, :, :]
            q = x_conv[:, -self.out_channels:, :, :]
            tc_out = (p + x_tca) * torch.sigmoid(q)
        elif self.activation == "sigmoid":
            tc_out = torch.sigmoid(x_conv)
        else:
            raise ValueError("No such activation")
        tc_out = tc_out.permute(0, 3, 2, 1)
        return tc_out


class GraphConv(nn.Module):
    """
        :param x: [batch, time_step, num_nodes, in_features].
        :param in_channels: int, size of input features
        :param out_channels: int, size of output features.
        :param approx: str, approximation method
        :param use_bias: bool, bias
        :return: tensor, [batch_size, time_step, num_nodes, in_features].
        """
    def __init__(self, in_channels: int, out_channels: int, approx: str, use_bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.approx = approx
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, self.out_channels))
        self.init_params()

    def init_params(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.xavier_uniform_(self.bias)

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        # Graph Convolution Approximation: 1. Linear 2. Chebshev
        gc_out = None
        if self.approx == "Linear":
            fully_conn = torch.matmul(x, self.weight)
            gc_out = torch.matmul(kernel, fully_conn)
        elif self.approx == "Cheb":
            gc_out = kernel

        # add bias
        if self.use_bias:
            gc_out += self.bias

        return gc_out


class Output(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ko: int, num_nodes: int):
        super(Output, self).__init__()
        self.temporal_conv1 = TemporalConv(kt=ko,
                                           in_channels=in_channels,
                                           out_channels=in_channels
                                           )
        self.layer_norm = nn.LayerNorm([num_nodes, in_channels])
        self.temporal_conv2 = TemporalConv(kt=1,
                                           in_channels=in_channels,
                                           out_channels=in_channels,
                                           activation="sigmoid"
                                           )
        self.linear = nn.Linear(in_features=in_channels,
                                out_features=out_channels,
                                bias=True
                                )

    def forward(self, x: torch.Tensor):
        output_t1 = self.temporal_conv1(x)
        output_ln = self.layer_norm(output_t1)
        output_t2 = self.temporal_conv2(output_ln)
        output = self.linear(output_t2)
        return output


class STConvBlock(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int,
                 temporal_channels: int, kt: int,
                 spacial_channels: int, graph_conv_approx: str):
        super(STConvBlock, self).__init__()
        self.tc_block1 = TemporalConv(kt=kt,
                                      in_channels=in_channels,
                                      out_channels=temporal_channels,
                                      activation="GLU")
        self.gc_block = GraphConv(in_channels=temporal_channels,
                                  out_channels=spacial_channels,
                                  approx=graph_conv_approx,
                                  use_bias=True)
        self.tc_block2 = TemporalConv(kt=kt,
                                      in_channels=spacial_channels,
                                      out_channels=temporal_channels,
                                      activation="GLU")
        self.layer_norm = nn.LayerNorm([num_nodes, temporal_channels])

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        out1 = self.tc_block1(x)
        out2 = torch.relu(self.gc_block(out1, kernel))
        out3 = self.tc_block2(out2)
        out4 = self.layer_norm(out3)
        return out4


class STGCN(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 graph_conv_kernel: torch.Tensor,
                 num_features: int = 1,
                 num_timesteps_input: int = 12,
                 num_timesteps_output: int = 1,
                 temporal_channels: int = 64,
                 spacial_channels: int = 16,
                 graph_conv_approx: str = "Linear",
                 temporal_kernel_size: int = 3):
        super(STGCN, self).__init__()
        self.graph_conv_kernel = graph_conv_kernel
        self.st_conv_block1 = STConvBlock(num_nodes=num_nodes,
                                          in_channels=num_features,
                                          temporal_channels=temporal_channels,
                                          kt=temporal_kernel_size,
                                          spacial_channels=spacial_channels,
                                          graph_conv_approx=graph_conv_approx
                                          )
        self.st_conv_block2 = STConvBlock(num_nodes=num_nodes,
                                          in_channels=temporal_channels,
                                          temporal_channels=temporal_channels,
                                          kt=temporal_kernel_size,
                                          spacial_channels=spacial_channels,
                                          graph_conv_approx=graph_conv_approx
                                          )
        self.ko = num_timesteps_input - 2 * 2 * (temporal_kernel_size - 1)
        if self.ko < 1:
            raise ValueError("temporal kernel size must be greater than 1, but received {}".format(self.ko))
        else:
            self.output = Output(in_channels=temporal_channels,
                                 out_channels=num_timesteps_output,
                                 ko=self.ko,
                                 num_nodes=num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_block1 = self.st_conv_block1(x, self.graph_conv_kernel)
        output_block2 = self.st_conv_block2(output_block1, self.graph_conv_kernel)
        output = self.output(output_block2)
        return output
