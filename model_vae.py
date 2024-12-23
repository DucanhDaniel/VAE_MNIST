import torch 
import torch.nn as nn

# Define custom PyTorch module
####
# Use after a fully connected (linear) layer to reshape a 
# flattened tensor back into a 3D (or higher) shape 
class Reshape(nn.Module):
    # Initialize the module and store desired shape in the self.shape attibute
    # The *args allows for passing a variable number of dimension
    # e.g. reshape_layer = Reshape(-1, 64, 7, 7)
    def __init__(self, *args):
        super().__init__()
        self.shape = args
        
    # Takes a tensor x as input and reshapes it to the specified shape
    def forward(self, x):
        # x.view(self.shape) is a function used for reshaping the tensor
        return x.view(self.shape)
 
# Use to trim tensors
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()
    
    # : means to keep all elements, e.g. batch and latent dimensions.
    def forward(self, x):
        return x[:, :, :28, :28]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: Shape(batch_size, latent_dim, [28 x 28])
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride = (1, 1), kernel_size = (3, 3), padding = 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride = (2, 2), kernel_size = (3, 3), padding = 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride = (2, 2), kernel_size = (3, 3), padding = 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride = (1, 1), kernel_size = (3, 3), padding = 1),
            nn.Flatten()
        )
        
        # 3136 = 64 * 7 * 7
        self.z_mean = torch.nn.Linear(in_features = 3136, out_features = 2)
        self.z_log_var = torch.nn.Linear(in_features= 3136, out_features= 2)
        
        self.decoder = nn.Sequential(
            # 2: Latent dimension
            nn.Linear(2, 3136),
            # -1: Auto correct batch size
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride = (1, 1), kernel_size = (3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride = (2, 2), kernel_size = (3, 3), padding = 1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride = (2, 2), kernel_size = (3, 3), padding = 0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride = (1, 1), kernel_size = (3, 3), padding = 0),
            Trim(),
            nn.Sigmoid()
        )
    def encoding_function(self, x):
        x = self.encoder(x)
        # Get tensor z_mean and log_var from nn.Linear module using x from encoder
        z_mean_tensor, z_log_var_tensor = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterization_trick(z_mean_tensor, z_log_var_tensor)
        return encoded
    
    def reparameterization_trick(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(z_mean.get_device())
        encoded = z_mean + eps * torch.exp(z_log_var/2.)
        return encoded
    
    
    
    def forward(self, x):
        x = self.encoder(x)
        z_mean_tensor, z_log_var_tensor = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterization_trick(z_mean_tensor, z_log_var_tensor)
        
        # encoded is a 2-dimensional tensor
        # encoded.size(0) is batch size
        # encoded.size(1) is latent space's total dimension
        
#     encoded = torch.tensor([
#     [1.2, -0.3],   # Encoded representation of sample 1
#     [0.5,  2.1],   # Encoded representation of sample 2
#     [-1.0, 0.8],   # Encoded representation of sample 3
#     [0.0, -1.5],   # Encoded representation of sample 4
#     ])
        decoded = self.decoder(encoded)
        return encoded, z_mean_tensor, z_log_var_tensor, decoded
    