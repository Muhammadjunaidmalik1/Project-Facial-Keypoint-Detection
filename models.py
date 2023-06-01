## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # (W-F)/S=(224-5)1+1=220, pooling 110x110, 32 channels
        self.conv1 = nn.Conv2d(1, 32, 5)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # output tensor (32, 110, 110)
        self.fc_drop1 = nn.Dropout(p=0.2)
        
        
        
        # (110-5)/1+1=106
        self.conv2 = nn.Conv2d(32, 36, 5)
        # output (24, 106,106)
        # max pulling: (24,53,53)
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_drop2 = nn.Dropout(p=0.2)
        
                
        # 
        self.conv3 = nn.Conv2d(36, 48, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc_drop3 = nn.Dropout(p=0.2)
        
        # 
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc_drop4 = nn.Dropout(p=0.2)
        

        self.conv5 = nn.Conv2d(64, 64, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc6 = nn.Linear(64*4*4, 136)
       

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #x = self.dropout1(self.maxpool1(self.relu1(self.batchnorm1(self.conv1(x)))))
        #x = self.dropout2(self.maxpool2(self.relu2(self.batchnorm2(self.conv2(x)))))
        #x = self.dropout3(self.maxpool3(self.relu3(self.batchnorm3(self.conv3(x)))))
        
        #x = self.maxpool1(self.relu1(self.conv1(x)))
        #x = self.maxpool2(self.relu2(self.conv2(x)))
        #x = self.maxpool3(self.relu3(self.conv3(x)))
        #x = x.view(x.size(0), -1)
        #x = self.dropout(self.relu(self.linear1(x)))
        #x = self.relu(self.linear2(x))

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.fc_drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.fc_drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.fc_drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.fc_drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))

        
        x = x.view(x.size(0), -1)

        x = self.fc6(x)

        return x
