from __future__ import print_function
from __future__ import division
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch import autograd
from torch.autograd import Variable
import scipy.misc
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import pickle as pkl
import pickle
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import tqdm
from tqdm import tqdm


# In[3]:


torch.cuda.is_available()


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


data_file = "../data/train_data/train_task_1_2.csv"
log_file = "../logs/task_1_log.txt"
batch_size = 32
lr = 0.0001
num_epochs = 100
K = 100
total_q = 28000
total_s = 119000


# In[ ]:


f = open(log_file,'a')


# In[6]:


# df = pd.read_csv(data_file)


# In[7]:


# df.head()


# In[8]:


# df.describe()


# In[9]:


# df['QuestionId'].max()


# In[10]:


# df['UserId'].max()


# In[11]:


class Question_Ans(Dataset):
    def __init__(self, filename, mode='train'):
        self.df = pd.read_csv(filename)
        self.questionid = self.df['QuestionId'].values
        self.userid = self.df['UserId'].values
        self.ans = self.df['IsCorrect'].values
        
        self.ans = 2*self.ans - 1
        
        self.length=len(self.ans)
        
        
        if(mode=='train'):
            start=int(0*self.length)
            end=int(0.8*self.length)
        else:
            start=int(0.8*self.length)
            end=int(1*self.length)
            
            
        self.questionid = self.questionid[start:end]
        self.userid = self.userid[start:end]
        self.ans = self.ans[start:end]
        
        self.length=len(self.ans)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        qid = self.questionid[idx]
        uid = self.userid[idx]
        ans = self.ans[idx]
        return qid,uid,ans


# In[12]:


train_dataset = Question_Ans(filename=data_file,mode='train')
val_dataset = Question_Ans(filename=data_file,mode='val')


# In[13]:


train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=False)


# In[14]:


dataloader = {}
dataloader['train'] = train_dataloader
dataloader['val'] = val_dataloader


# In[15]:


dtype=torch.FloatTensor


# In[16]:


# Declare two matrices 
# Q = shape (total_q,K)
# U = shape (total_s,K)
Q = torch.nn.Embedding(total_q,K)
U = torch.nn.Embedding(total_s,K)

Q = Q.to(device)
U = U.to(device)


# In[17]:


# embedding = torch.nn.Embedding(5,8)


# In[18]:


# embedding(torch.LongTensor(np.array([3,4,3])))


# In[19]:


def get_qvector(questions):
    
#     q_list = []
#     for i in range(questions.shape[0]):
#         q_list.append(Q[i])
    
#     return torch.cat(q_list,dim=0)
    ans = Q(torch.LongTensor(questions).to(device))
    
    ans = ans.to(device)
    return ans


# In[20]:


def get_uvector(users):
    
#     u_list = []
#     for i in range(users.shape[0]):
#         u_list.append(U[i])
    
#     return torch.cat(u_list,dim=0)
#     print(users)
    ans = U(torch.LongTensor(users).to(device))
    ans = ans.to(device)
    return ans


# In[21]:


test1 = torch.randn(32,100)
test2 = torch.randn(32,100)
test1 = torch.unsqueeze(test1,1)
test2 = torch.unsqueeze(test2,2)
print(test1.shape)
print(test2.shape)
res = torch.bmm(test1,test2)
print(res.shape)
res = torch.squeeze(res)
print(res.shape)


# In[22]:


def get_score(qvectors,uvectors):
    
    q_unsq = torch.unsqueeze(qvectors, 1)
    u_unsq = torch.unsqueeze(uvectors, 2)
    score = torch.bmm(q_unsq,u_unsq)
    score = torch.squeeze(score)
    return score


# In[23]:


params_to_update = list(Q.parameters()) + list(U.parameters())


# In[24]:


criterion = nn.MSELoss()
optimizer = optim.SGD(params=params_to_update,lr=lr, momentum=0.9)


# In[ ]:


best_loss = 1e9
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        
            
        running_loss = 0.0
        
        # Iterate over data.
        cnt = 0
#         print("*********** entered ",phase,"*********************")
        for questions,users,ans in tqdm(dataloader[phase]):
            
#             print("************ loaded data of one batch*******")
            
            cnt = cnt + 1
            ans = torch.tensor(ans)
            ans = ans.type(dtype)
            ans = ans.to(device)
            
            qvectors = get_qvector(questions)
            uvectors = get_uvector(users)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            
            scores = get_score(qvectors,uvectors)
            
            loss = criterion(scores,ans)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
        
            # statistics
            running_loss += loss.item()
            
            # compute correct
            # running_corrects += torch.sum(preds == labels.data)
            
            if (cnt % 1000 == 0 ) :
                print('Batch {} {} Loss: {:.4f} '.format(str(cnt),phase, running_loss/(cnt*batch_size)))
                print('Batch {} {} Loss: {:.4f} '.format(str(cnt),phase, running_loss/(cnt*batch_size)),file=f)
            

        epoch_loss = running_loss / len(dataloader[phase].dataset)

        print(' {} Loss: {:.4f} '.format(phase, epoch_loss))
        print(' {} Loss: {:.4f} '.format(phase, epoch_loss),file=f)
        
        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            
            q_file = '../Weights/simple_fact_q' + '_best' + '.pkl'
            u_file = '../Weights/simple_fact_u' + '_best' + '.pkl'
            q_f = open(q_file, 'wb')
            u_f = open(u_file, 'wb')
            pickle.dump(Q,q_f)
            pickle.dump(U,u_f)
        
        q_file = '../Weights/simple_fact_q_' + str(epoch+1) + '.pkl'
        u_file = '../Weights/simple_fact_u_' + str(epoch+1) + '.pkl'
        q_f = open(q_file, 'wb')
        u_f = open(u_file, 'wb')
        pickle.dump(Q,q_f)
        pickle.dump(U,u_f)
    print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Loss: {:4f}'.format(best_loss))

print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),file=f)
print('Best val Loss: {:4f}'.format(best_loss),file=f)

