from model import *

# hyper-params
spacial_channels = 16
temporal_channels = 64
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
num_nodes = 500
device = "cuda" if torch.cuda.is_available() else "cpu"


tensor = torch.randn(50, 12, num_nodes, 1).to(device)
kernel = torch.randn(num_nodes, num_nodes).to(device)
model = Model(num_nodes=num_nodes, num_features=1, graph_conv_kernel=kernel).to(device)
y = model(tensor)
print(list(y.shape))


# device, model, loss, optimizer
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Model().to(device)
# criterion = nn.MSELoss().to(device)
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#
#
# def train():
#     for epoch in range(epochs):
#         pass
