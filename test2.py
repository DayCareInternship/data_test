import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# load one dataset

ts_data = pd.read_csv('./ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_60.csv',index_col = 0)
#ts_data = ts_data.astype('float')
ts_data.head()

ts_data[['is_anomaly','value']].groupby('is_anomaly').count()

ts_data.shape

# define train, test and validation datasets

train_percent = int(0.3*len(ts_data))
valid_percent = int(0.1*len(ts_data))
test_percent = int(0.6*len(ts_data))

train_data = list(ts_data.iloc[:train_percent,0])
valid_data = list(ts_data.iloc[train_percent:train_percent+valid_percent,0])
test_data = list(ts_data.iloc[train_percent+valid_percent:,0])

# define parameters

w = 45
pred_window = 1
filter1_size = 128
filter2_size = 32
kernel_size = 2
stride = 1
pool_size = 2


# generate subsequences on which we will train

def get_subsequences(data):
    X = []
    Y = []

    for i in range(len(data) - w - pred_window):
        X.append(data[i:i + w])
        Y.append(data[i + w:i + w + pred_window])
    return np.array(X), np.array(Y)


trainX, trainY = get_subsequences(train_data)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

validX, validY = get_subsequences(valid_data)
validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

testX, testY = get_subsequences(test_data)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#  CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## layers of a CNN

        self.conv1 = nn.Conv1d(1, filter1_size, kernel_size, stride, padding=0)

        self.conv2 = nn.Conv1d(filter1_size, filter2_size, kernel_size, stride, padding=0)

        self.maxpool = nn.MaxPool1d(pool_size)

        self.dim1 = int(0.5 * (0.5 * (w - 1) - 1)) * filter2_size

        self.lin1 = nn.Linear(self.dim1, pred_window)
        # self.lin2 = nn.Linear(1000,pred_window)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # convolution layer 1
        x = (F.relu(self.conv1(x)))
        x = self.maxpool(x)
        # print(x.shape)
        # x = self.dropout(x)

        # convolution layer 2
        x = (F.relu(self.conv2(x)))
        x = self.maxpool(x)
        # x = self.dropout(x)

        # print(x.shape)

        # print(x.shape)
        # print(int(0.25* (w) * filter2_size))
        x = x.view(-1, self.dim1)

        x = self.dropout(x)
        x = self.lin1(x)
        # x = self.dropout(x)
        # x = self.lin2(x)

        return x

    # define CNN model

    model_A1 = Net()
    print(model_A1)

    # define optimizer

    criterion_scratch = nn.L1Loss()
    optimizer_scratch = optim.Adam(model_A1.parameters(), lr=1e-5, weight_decay=1e-6)

    # function for training the model (also checks on validation data)

    def train_valid(n_epochs, trainX, trainY, validX, validY, model, optimizer, criterion, save_path, freq=5):
        """returns trained model"""

        target_train = torch.tensor(trainY).type('torch.FloatTensor')
        data_train = torch.tensor(trainX).type('torch.FloatTensor')

        target_valid = torch.tensor(validY).type('torch.FloatTensor')
        data_valid = torch.tensor(validX).type('torch.FloatTensor')

        train_loss_min = np.Inf
        valid_loss_min = np.Inf
        last_valid_loss = 0

        for epoch in range(1, n_epochs + 1):

            ###################
            # training the model #
            ###################
            model.train()

            # print(data.shape)

            optimizer.zero_grad()
            output = model(data_train)
            loss = criterion(output, target_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            ###################
            # Validation #
            ###################
            model.eval()
            output_valid = model(data_valid)

            loss_valid = criterion(output_valid, target_valid)
            valid_loss = loss_valid.item()
            if (valid_loss == last_valid_loss):
                print('problem')

            last_valid_loss = valid_loss
            if (epoch % freq == 0):
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch,
                    train_loss,
                    valid_loss
                ))

            if valid_loss < valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss

        return model, output

    # train model

    model_A1, out = train_valid(500, trainX, trainY, validX, validY, model_A1, optimizer_scratch,
                                criterion_scratch, 'model_A1.pt', freq=10)

    # load best model (saved previously during training)

    model_A1.load_state_dict(torch.load('model_A1.pt'))

    # make predictions

    test_tensor = torch.tensor(testX).type('torch.FloatTensor')
    model_A1.eval()
    out = model_A1(test_tensor)
    out = out.detach().numpy()

    df_out = pd.DataFrame()
    df_out['pred'] = out[:, 0]
    df_out['actual'] = testY[:, 0]
    # df_out.index = ts_data.index[train_percent + valid_percent:len(ts_data)-w-pred_window]

    df_out.tail()



# compute error (actual - pred)

df_out['error'] = np.abs(df_out['pred'] - df_out['actual'])
df_out['error_n'] = (df_out['error'] - df_out['error'].mean())/df_out['error'].std()
df_out.index = ts_data.index[train_percent + valid_percent +w+pred_window-1:-1]


# check whether error is more than the threshold

thresh = df_out.loc[df_out['error_n'].abs() > 3]
thresh

# calc TP, FN, FP, TN

positives = ts_data.loc[df_out.index].loc[ts_data.is_anomaly == 1].index
negatives = ts_data.loc[df_out.index].loc[ts_data.is_anomaly == 0].index
tp = []
fn = []
fp = []
tn = []
for p in positives:
    if p in thresh.index:
        tp.append(p)
    else:
        fn.append(p)

for n in negatives:
    if n in thresh.index:
        fp.append(n)
    else:
        tn.append(n)

        calc
        F - score

    recall = len(tp) / (len(tp) + len(fn))
    precision = len(tp) / (len(tp) + len(fp))
    F_score = 2 * recall * precision / (recall + precision)
    F_score