from model import *

# hyper-params
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

# Device, Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train():
    for epoch in range(epochs):
        pass
