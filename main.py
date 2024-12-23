import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from helper_data import get_dataloaders_mnist
from helper_train import train_vae_v1
from helper_utils import set_deterministic, set_all_seeds
from helper_plotting import plot_training_loss
from helper_plotting import plot_generated_images
from helper_plotting import plot_latent_space_with_labels
from helper_plotting import plot_images_sampled_from_vae
from model_vae import VAE


if __name__ == "__main__":
    ##########################
    ### SETTINGS
    ##########################
    
    # Device
    CUDA_DEVICE_NUM = 0
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)
    
    # Hyperparameters
    RANDOM_SEED = 123
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    
    set_deterministic
    set_all_seeds(RANDOM_SEED)
    
    ##########################
    ### Dataset
    ##########################
    
    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
        batch_size=BATCH_SIZE, 
        num_workers=2, 
        validation_fraction=0.)
    
    # Checking the dataset
    print('Training Set:\n')
    for images, labels in train_loader:  
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
        
    # Checking the dataset
    print('\nValidation Set:')
    for images, labels in valid_loader:  
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    
    # Checking the dataset
    print('\nTesting Set:')
    for images, labels in test_loader:  
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    
    set_all_seeds(RANDOM_SEED)
    
    model = VAE()
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    log_dict = train_vae_v1(num_epochs=NUM_EPOCHS, model=model, 
                        optimizer=optimizer, device=DEVICE, 
                        train_loader=train_loader,
                        skip_epoch_stats=True,
                        logging_interval=50)
    
    
    plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], NUM_EPOCHS, custom_label=" (reconstruction)")
    plot_training_loss(log_dict['train_kl_loss_per_batch'], NUM_EPOCHS, custom_label=" (KL)")
    plot_training_loss(log_dict['train_combined_loss_per_batch'], NUM_EPOCHS, custom_label=" (combined)")
    plt.show()
    
    plot_generated_images(data_loader=train_loader, model=model, device=DEVICE, modeltype='VAE')           

    plot_latent_space_with_labels(
    num_classes=10,
    data_loader=train_loader,
    encoding_fn=model.encoding_function, 
    device=DEVICE)

    plt.legend()
    plt.show()
    