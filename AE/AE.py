"""
This code mainly follows the Geeks4Geeks Pytorch Autoencoder example.
https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

Any modifiations are made by the AABL Lab.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

class AE(nn.Module):
    def __init__(self, input_dim, learning_rate=1e-3):
        super().__init__()

        self.input_dim = input_dim # Input dimensions of data

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim//2)),
            nn.ReLU(),
            nn.Linear((input_dim//2), (input_dim//4)),
            nn.ReLU(),
            nn.Linear((input_dim//4), (input_dim//8)),
            nn.ReLU(),
            nn.Linear((input_dim//8), (input_dim//16)),
            nn.ReLU(),
            nn.Linear((input_dim//16), (input_dim//32)),
            nn.ReLU(),
            nn.Linear((input_dim//32), (input_dim//64))
        )

        self.decoder = nn.Sequential(
            nn.Linear((input_dim//64), (input_dim//32)),
            nn.ReLU(),
            nn.Linear((input_dim//32), (input_dim//16)),
            nn.ReLU(),
            nn.Linear((input_dim//16), (input_dim//8)),
            nn.ReLU(),
            nn.Linear((input_dim//8), (input_dim//4)),
            nn.ReLU(),
            nn.Linear((input_dim//4),  (input_dim//2)),
            nn.ReLU(),
            nn.Linear((input_dim//2), input_dim),
            nn.Sigmoid()
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
                # Reshape mini-batch data to [N, 784] matrix
                batch_data = batch_data.view(-1, self.input_dim)
                
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
    model = AE(input_dim=784) # Dimensions of MNIST images flattened (28 x 28 = 784)

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
