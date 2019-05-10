import torch 

class CNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(CNN, self).__init__()
        
        #Input channels = 3, output channels = 64
        self.conv1_1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 16 x 256 x 256
        self.conv_1_2 = torch.nn.Conv2d(16,16, kernel_size=3,stride=1,padding=1)
        # 16 x 256 x 256

        # self.bn1_1 = nn.BatchNorm2d(16)
        # self.bn1_2 = nn.BatchNorm2d(16)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 16 x 128 x 128

        #Input channels = 16, output channels = 32
        self.conv2_1 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 32 x 128 x 128
        self.conv_2_2 = torch.nn.Conv2d(32,32, kernel_size=3,stride=1,padding=1)
        # 32 x 128 x 128
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 32 x 64 x 64

        #Input channels = 64, output channels = 256
        self.conv3_1 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 64 x 64 x 64
        self.conv_3_2 = torch.nn.Conv2d(64,64, kernel_size=3,stride=1,padding=1)
        # 64 x 64 x 64
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 64 x 32 x 32

        # input features, 6400 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 4096)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(4096, 1000)

        self.fc3 = torch.nn.Linear(1000, 10)

        self.activation_F = torch.nn.ReLU(True)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        layer_1 = self.pool1(self.activation_F(self.conv_1_2(self.activation_F(self.conv1_1(x)))))
        layer_2 = self.pool2(self.activation_F(self.conv_2_2(self.activation_F(self.conv2_1(layer_1)))))
        layer_3 = self.pool3(self.activation_F(self.conv_3_2(self.activation_F(self.conv3_1(layer_2)))))

        flat = layer_3.view(-1,64*16*16  )
        fc1  = self.activation_F(self.fc1(flat))
        fc2  = self.activation_F(self.fc2(fc1))
        fc3  = self.fc3(fc2)

        return fc3



