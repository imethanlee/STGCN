from stgcn import *
from data.loader import *
from torch.utils.data import DataLoader
from sklearn import metrics
from torchsummary import summary
import tqdm
import time

# config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cwd = "."
v_path = cwd + "/data/PeMS_V_228.csv"
w_path = cwd + "/data/PeMS_W_228.csv"
print("GPU: ", torch.cuda.get_device_name(0))
print("PyTorch Version: ", torch.__version__)

# Data setup
day_train = 34
day_val = 5
day_slot = int(60 / 5 * 24)
tfd = TrafficFlowData(device, v_path, w_path, day_train * day_slot, day_val * day_slot)


# Model Initialization
learning_rate = 1e-4
weight_decay = 1e-6
epochs = 50
batch_size = 32
num_nodes = tfd.num_nodes


train_iter = DataLoader(tfd.train, batch_size, shuffle=False)
val_iter = DataLoader(tfd.val, batch_size, shuffle=False)
test_iter = DataLoader(tfd.test, batch_size, shuffle=False)


model = STGCN(num_nodes=num_nodes, graph_conv_kernel=tfd.get_conv_kernel("Linear")).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99999)


def val():
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x).view(len(x), -1)
            loss = criterion(y_pred, y)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        return loss_sum / n


def train():
    val_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0., 0
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val()

        print('Epoch: {:03d} | Lr: {:.20f} | Train loss: {:.6f} | Val loss: {:.6f}'.format(
            epoch, optimizer.param_groups[0]['lr'], loss_sum / n, val_loss))
        time.sleep(0.2)
        val_losses.append(val_loss)
    print("Training Completed!")


def test():
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in test_iter:
            y_pred = model(x).view(len(x), -1)
            l = criterion(y_pred, y)
            loss_sum += l.item() * y.shape[0]
            n += y.shape[0]
        loss = loss_sum / n

        mae, mape, mse = [], [], []
        for x, y in test_iter:
            y = Utils.inverse_z_score(y.cpu().numpy()).reshape(-1)
            y_pred = Utils.inverse_z_score(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
    print('Test loss {:.6f}'.format(loss))
    print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(MAE, MAPE, RMSE))


summary(model, (12, num_nodes, 1))

train()
test()



