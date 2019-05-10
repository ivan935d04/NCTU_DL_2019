from prepro import * 
from model import * 
import torch 
import pandas as pd 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import tqdm
from torchvision import transforms, datasets
#load data
# Data = AllData(root='./animal-10/train/',mode='restore',name='img_path.p',transform=transforms.Compose([
#                                                Rescale(130),
#                                                RandomCrop(128),
#                                                ToTensor()]))

# valData = AllData(root='./animal-10/val/',mode='create',name='val_img_path.p',transform=transforms.Compose([
#                                                Rescale(130),
#                                                RandomCrop(128),
#                                                ToTensor()]))
data_transform = transforms.Compose([
        transforms.Resize(130),
        transforms.RandomCrop(128),
        transforms.ToTensor(),
    ])
train_dataset = datasets.ImageFolder(root='animal-10/train',
                                           transform=data_transform)
dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=100, shuffle=True,
                                             num_workers=4)

val_dataset = datasets.ImageFolder(root='animal-10/val',
                                           transform=data_transform)
valdataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=100, shuffle=True,
                                             num_workers=0)

# valdataloader = DataLoader(valData, batch_size=100,
#                         shuffle=True, num_workers=0)
# dataloader = DataLoader(Data, batch_size=100,
#                         shuffle=True, num_workers=4)

model = CNN()
epoch = 50
lr = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=lr) 
criterion = torch.nn.CrossEntropyLoss()
model = model.train()
loss_rec = []
loss_val_rec = []
step_rec = []
acc_rec = []
acc_val_rec = []
step = 0
for i in range(epoch):
    if (epoch >= 5) & ((epoch+1) % 10 ==0):
        torch.save(model.state_dict(), './CNN_epoch'+str(epoch))
    for i_batch, sample_batched in tqdm.tqdm(enumerate(dataloader)):

        current_loss = 0
        current_correct = 0
        optim.zero_grad()
        # print(sample_batched['image'])
        output = model.forward(sample_batched[0].float())
        # print(sample_batched[0].size())
        # print(output.size())
        loss = criterion(output, sample_batched[1])
        loss.backward()
        optim.step()
        if (step % 50 == 0):
            loss_rec += [loss.item()]
            _, preds = torch.max(output,1)
            acc_rec += [torch.sum(preds == sample_batched[1]).double()/100]
            val_loss = 0
            val_correct = 0
            for i_b, val_data in enumerate(valdataloader):
                val_output = model.forward(val_data[0].float())
                _, val_preds = torch.max(val_output, 1)
                val_loss += criterion(output, val_data[1])
                val_correct += torch.sum(val_preds == val_data[1].data)
            loss_val_rec += [val_loss/40]
            step_rec += [step]
            acc_val_rec += [val_correct/4000]
            print('batch = {}'.format(i_batch), 'val_batch_loss={}'.format(val_loss/40), 'val_batch_acc={}'.format(val_correct.double()/4000))
            print('batch = {}'.format(i_batch), 'batch_loss={}'.format(loss.item()), 'batch_acc={}'.format(torch.sum(preds == sample_batched[1]).double()/100))
        current_loss += loss.item()*sample_batched[0].size(0)
        current_correct += torch.sum(preds == sample_batched[1].data)
        step += 1
    epoch_loss = current_loss / 100
    epoch_acc = current_correct.double() /  100
    print('epoch = {}'.format(i), 'epoch_loss={}'.format(epoch_loss), 'epoch_acc={}'.format(epoch_acc))
    if ((epoch+1) % 5 == 0):
        pickle.dump((loss_rec,acc_rec),open('train_loss_acc_ep_'+str(epoch)+'.p','wb'))
        pickle.dump((loss_val_rec,acc_val_rec),open('test_loss_acc_ep_'+str(epoch)+'.p','wb'))

    
    