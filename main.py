from model.stgcn import *
from data.loader import *
from torch.utils.data import DataLoader
from torchsummary import summary
import tqdm
import time
import argparse

# arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=int, default=1e-3)
parser.add_argument('--weight_decay', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--drop_rate', type=float, default=0.5)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--approx', type=str, default='linear')
parser.add_argument('--v_path', type=str, default='./data/metr_V_207.csv')
parser.add_argument('--w_path', type=str, default='./data/metr_W_207.csv')
parser.add_argument('--save_path', type=str, default='./model/save/')
parser.add_argument('--in_timesteps', type=int, default=12)
parser.add_argument('--out_timesteps', type=int, default=3)
parser.add_argument('--train_days', type=int, default=34)
parser.add_argument('--val_days', type=int, default=5)
parser.add_argument('--test_days', type=int, default=5)

args = parser.parse_args()

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
    print("GPU: ", torch.cuda.get_device_name(0))
print("PyTorch Version: ", torch.__version__)

# Data setup
data = TrafficFlowData(device=device,
                       v_path=args.v_path,
                       w_path=args.w_path,
                       len_train=args.train_days * 288,
                       len_val=args.val_days * 288,
                       in_timesteps=args.in_timesteps,
                       out_timesteps=args.out_timesteps)
train_iter = DataLoader(data.train, args.batch_size)
val_iter = DataLoader(data.val, args.batch_size)
test_iter = DataLoader(data.test, args.batch_size)

# Model Initialization
model = STGCN(num_nodes=data.num_nodes, graph_conv_kernel=data.get_conv_kernel("Linear")).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


def val():
    model.eval()
    loss_sum, n = 0., 0
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x).view(len(x), -1)
            loss = criterion(y_pred, y)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        return loss_sum / n


def train():
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0., 0
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val()
        print('Epoch: {:03d} | Lr: {:.20f} | Train loss: {:.6f} | Val loss: {:.6f}'.format(
            epoch, optimizer.param_groups[0]['lr'], loss_sum / n, val_loss))
        scheduler.step()
        torch.save(model.state_dict(), './model/save/stgcn_{}'.format(args.approx))
    print("Training Completed!")


def test():
    model.eval()
    loss_sum, n = 0., 0
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
            mape += (np.divide(d, y)).tolist()
            mse += (np.power(d, 2)).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
    print('Test loss {:.6f}'.format(loss))
    print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(MAE, MAPE, RMSE))


# summary(model, (12, num_nodes, 1))
train()
test()



