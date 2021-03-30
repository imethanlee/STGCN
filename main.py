from model.stgcn import *
from data.loader import *
from utils.early_stop import *
from torch.utils.data import DataLoader
from torchsummary import summary
import tqdm
import argparse

# arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=int, default=1e-3)
parser.add_argument('--weight_decay', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--decay_rate', type=float, default=0.7)
parser.add_argument('--decay_steps', type=int, default=5)
parser.add_argument('--approx', type=str, default='linear')
parser.add_argument('--save_path', type=str, default='./model/save/')
parser.add_argument('--patience', type=int, default=20)

parser.add_argument('--v_path', type=str, default='./data/v_pems_228.csv')
parser.add_argument('--w_path', type=str, default='./data/w_pems_228.csv')
parser.add_argument('--n_time', type=int, default=12)
parser.add_argument('--out_time', type=int, default=3)
parser.add_argument('--train_pct', type=float, default=0.7)
parser.add_argument('--test_pct', type=float, default=0.2)

args = parser.parse_args()
model_save_path = args.save_path + "STGCN_trained_" + args.v_path[-12:-4] + "_" + str(args.out_time)

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
    print("GPU: ", torch.cuda.get_device_name(0))
print("PyTorch Version: ", torch.__version__)

# Data setup
data = TrafficFlowData(device=device,
                       v_path=args.v_path,
                       w_path=args.w_path,
                       train_pct=args.train_pct,
                       test_pct=args.test_pct,
                       n_time=args.n_time,
                       out_time=args.out_time)
train_iter = DataLoader(data.train, args.batch_size)
val_iter = DataLoader(data.val, args.batch_size)
test_iter = DataLoader(data.test, args.batch_size)

# Model Initialization
model = STGCN(num_nodes=data.n_node, graph_conv_kernel=data.get_conv_kernel("linear")).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)


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
    early_stop = EarlyStop(args.patience)
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
        scheduler.step()

        val_loss = val()

        if early_stop.check(val_loss):
            break
        if early_stop.save:
            torch.save(model.state_dict(), model_save_path)

        print('Epoch: {:03d} | Lr: {:.20f} | Train loss: {:.6f} | Val loss: {:.6f} | Early Stop: {:02d}'.format(
            epoch, optimizer.param_groups[0]['lr'], loss_sum / n, val_loss, early_stop.cnt))

    print("Training Completed!")


def test():
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    loss_sum, n = 0., 0
    with torch.no_grad():
        for x, y in test_iter:
            y_pred = model(x).view(len(x), -1)
            l = criterion(y_pred, y)
            loss_sum += l.item() * y.shape[0]
            n += y.shape[0]
        loss = loss_sum / n

        ae, ape, se = [], [], []
        for x, y in test_iter:
            y = Utils.inverse_z_score(y.cpu().numpy()).reshape(-1)
            y_pred = Utils.inverse_z_score(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            ae += d.tolist()
            ape += (np.divide(d, y)).tolist()
            se += (np.power(d, 2)).tolist()
        mae = np.array(ae).mean()
        mape = np.array(ape).mean()
        rmse = np.sqrt(np.array(se).mean())
    print('Test loss {:.6f}'.format(loss))
    print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(mae, mape, rmse))


# summary(model, (12, num_nodes, 1))
train()
test()



