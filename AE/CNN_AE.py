"""
This code mainly follows the Geeks4Geeks Pytorch Autoencoder example.
https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

Any modifiations are made by the AABL Lab.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

class CNN_AE(nn.Module):

    def __init__(self, input_dim, channels=1, learning_rate=1e-3):
        super().__init__()
        #TODO: It might be good to be able to calculate reduction till it hits desired latent space
        # Perhaps in a loop, etc. but for the sake of initial implementation I have not done this.

        self.input_dim = input_dim # Input dimensions of data
        self.channels = channels # Black and white images channel = 1, rgb = 3

        #This "fun" math note can be moved to a readme later
        '''
        NOTE:
        A fun *math* reminder about convolutional layer dimensions!
        Giving this note since the current layout may not work well for you! 
        And I want to save you heartache!

        The output dimension of a 2D Convolution is equal to the formula:
        Floor[((W-K+2P)/S) + 1]
        Where:
        W = Input Size (MNIST width is 28)
        K = Kernal Size
        P = Padding
        S = Stride

        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        The output dimension of a MaxPool 2D is equal to the formula:
        Floor[((W+2P-D*(K-1)-1)/S)+1]
        Where:
        W = Input Size 
        P = Padding
        D = Dilation
        K = Kernal Size
        S = Stride

        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html 

        The output dimension of a 2D Transpose Convolution is equal to the formula:
        (W-1)*S-2*P+D*(K-1)+O+1
        Where:
        W = Input Size 
        S = Stride
        P = Padding
        D = Dilation
        K = Kernal Size
        O = Output Padding

        Also note that channels just change to what you set them to.
        ''' 

        self.encoder = nn.Sequential(
            #I will run through MNIST dataset as example
            # Input: 28 x 28 x 1 (the 1 in question is the number of channels)
            nn.Conv2d(channels, input_dim//2, 3, stride=3, padding=1),  # Input -> Floor[((28-3+2(1))/3)+1] = 10 x 10 x (input_dim = 28 // 2) = 14
            nn.ReLU(True), # 10 x 10 x 14
            nn.MaxPool2d(2, stride=2),  # 10 x 10 x 14 -> Floor[(10+(2*0)-1*(2-1)-1)/2)+1] = 5 x 5 x 14
            nn.Conv2d(input_dim//2, input_dim//4, 3, stride=2, padding=1),  # 5 x 5 x 14 -> Floor[((5-3+2)/2)+1] 3 x 3 x 7
            nn.ReLU(True), # 3 x 3 x 7
            nn.MaxPool2d(2, stride=1) # 3 x 3 x 7 -> Floor[((3+(2*0)-1*(2-1)-1)/1)+1] = 2 x 2 x 7
        )

        self.decoder = nn.Sequential(
            # Input: 2 x 2 x 7
            nn.ConvTranspose2d(input_dim//4, input_dim//2, 3, stride=2),  # Input -> (2-1)*2-2*0+1*(3-1)+0+1 = 5 x 5 x 14
            nn.ReLU(True), # 5 x 5 x 14
            nn.ConvTranspose2d(input_dim//2, input_dim//4, 5, stride=3, padding=1),  # 5 x 5 x 14 -> 15 x 15 x 7
            nn.ReLU(True),
            nn.ConvTranspose2d(input_dim//4, channels, 2, stride=2, padding=1),  # 15 x 15 x 7 -> 28 x 28 x 1
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, data_loader, epochs, print_epoch_loss=True):
        for epoch in range(epochs):
            loss = 0
            for batch_data, _ in data_loader:
                
                self.optimizer.zero_grad() # Zero gradient for training
                
                outputs = self.forward(batch_data) # Feed forwards and get outputs
                
                train_loss = self.criterion(outputs, batch_data) # Compute loss
                train_loss.backward() # Compute backprop gradients from loss
                
                self.optimizer.step() # Update with computed gradients
                
                # Add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            
            # Compute the epoch training loss
            loss = loss / len(train_loader)
            
            # Display the epoch training loss
            if print_epoch_loss:
                print("Epoch : {}/{}, Loss = {:.6f}".format(epoch + 1, epochs, loss))
    

if __name__ == "__main__":
    model = CNN_AE(input_dim=26) # Assume square dimensions for MNIST this is 28 x 28

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    model.train(train_loader, 100)
