# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:45:48 2021

@author: fatca
"""
from rdkit import Chem as ch
from rdkit.Chem import AllChem as ach
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv
import torch.optim as optim


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & val loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['val'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['val'], c='tab:cyan', label='val')
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

class ECFPDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        with open(mode+'.csv', 'r') as fi:
            reader=csv.reader(fi)
            header=next(reader)
            y = [] 
            x = []
            for line in reader:
                m = ch.MolFromSmiles(line[0])
                try:
                    #用m to generate a radius of 2 adn 512 bit
                    fp = ach.GetMorganFingerprintAsBitVect(m, 2, nBits=512)
                except:
                    #假設上面error就填零 in 512 bit
                    fp = torch.zeros(512)
                    #save the result vector of fp in x
                x.append([float(i) for i in fp])
                if self.mode in ['train', 'val']:
                    #save target value in y
                    y.append(float(line[1]))
        #save target and data to pytorch form
        self.data = torch.tensor(x)
        self.target = torch.tensor(y)

        print('Finished reading the {} set of ECFP/Hf298 Dataset ({} samples found)'.format(mode, len(self.data)))
    #allow us to access individual samples in dataset by index
    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]
    #allow us to access total number samples of dataset
    def __len__(self):
        return len(self.data)

#create a pytorch dataloader object with specific mode and batchsize
def prep_dataloader(mode, batch_size):
    dataset = ECFPDataset(mode)
    #Dataloader is a pytorch object
    #if dataset mode is train shuffle is true
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'))
    return dataloader

#NN class inherits from all the funtions of pytorch 'nn.Module' class
class NeuralNetwork(nn.Module):
    def __init__(self):
        #call constructor of nn.Module class, ？？
        super(NeuralNetwork, self).__init__()
        
        #nn.Sequential represents the structure of nn, consist of 3 layers
        self.network = nn.Sequential(
            # full connected layers with 512 input n and 5 output n
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Linear(64,1)
        )
        #pytorch loss function 
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        #self.network(x) applies the neural network to the input tensor x, producing a PyTorch tensor as output

        #sqeeze(1) remove the dimension with size1 from the output tensor, this is necessary (cont.)
        #the nn produces ouput a tensor with shape(batch_size, 1) and return a tensor with shape (batch_size)
        return self.network(x).squeeze(1)

    def cal_loss(self, pred, target):
        #compute mse between pred and target and compute loss as pytorch tensor
        return self.criterion(pred, target)


def evaluate(data, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in data:                         # iterate through the dataloader
        # move data to device (cpu/cuda)
        x, y = x.to(device), y.to(device)

        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    # compute averaged loss
    total_loss = total_loss / len(data.dataset)
    return total_loss

#TODO: data feature selection (not neccessary)

def train(tr_set, val_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), lr=config['learning_rate'])
    #initualize the variable
    min_mse = 10000
    loss_record = {'train': [], 'val': []}
    early_stop_cnt = 0
    epoch = 0
    #loop for main training loop
    while epoch < n_epochs:
        model.train()
        loss = 0
        batchs = 0
        for x, y in tr_set:
            #clear the gradients of all model parameters
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            #computes the gradients of mse loss with respect to all parameters 
            mse_loss.backward()
            #update the parameters using computed gradients and optimized algorithm
            optimizer.step()
            loss += mse_loss
            batchs += 1

        loss_record['train'].append(evaluate(tr_set, model, device))
        loss_record['val'].append(evaluate(val_set, model, device))
        #for early stopping based on the validation loss
        #if vad mse lower than min_mse using torch.save() for saved to disk and early stop reset to 0
        if loss_record['val'][-1] < min_mse:
            min_mse = loss_record['val'][-1]
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch+1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        # if earlystopcount > specific setting number training is stopped early
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def predict(data, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in data:                            # iterate through the dataloader
        # move data to device (cpu/cuda)
        x = x.to(device)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    # concatenate all predictions and convert to a numpy array
    preds = torch.cat(preds, dim=0).numpy()
    return preds


############### main function #################

# get the current available device ('cpu' or 'cuda')
device = 'cpu'

# The trained model will be saved to ./models/
os.makedirs('models', exist_ok=True)
model = NeuralNetwork().to(device)
# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size':256,               # mini-batch size for dataloader
    # optimization algorithm (optimizer in torch.optim)
    'optimizer':'Adam',
    'learning_rate':0.0001,
    # early stopping epochs (the number epochs since your model's last improvement)
    'early_stop': 50,
    'save_path': 'models/model.pth'  # your model will be saved here
}

#TODO: load dataframe and spit to tr val test
 


tr_set = prep_dataloader('train', config['batch_size'])
val_set = prep_dataloader('val', config['batch_size'])
test_set = prep_dataloader('test', config['batch_size'])

model = NeuralNetwork().to(device)  # Construct model and move to device
model_loss, model_loss_record = train(tr_set, val_set, model, config, device)
plot_learning_curve(model_loss_record, title='deep model')
del model

# Load your best model
model = NeuralNetwork().to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)


# predict testing data and save predictions
print('Predicting Hf298 of testing data')
Hf_test = predict(test_set, model, device)
result=[]
with open('test.csv','r') as f, open('DNN_submission.csv','w',newline='') as g:
    reader=csv.reader(f)
    writer=csv.writer(g)
    next(reader)
    writer.writerow(['SMILES','Hf'])
    for i,line in enumerate(reader):
        line.append(Hf_test[i])
        result.append(line)
    writer.writerows(result)
        
print('****** Thank you for your hard work and dedication ******')
