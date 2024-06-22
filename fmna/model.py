import torch 

class AE_lite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Conv1d(50, 100, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(100, 50, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(50, 50, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Conv1d(50, 100, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Conv1d(100, 200, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Conv1d(200, 400, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(400, 200, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose1d(200, 100, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose1d(100, 50, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose1d(50, 50, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
