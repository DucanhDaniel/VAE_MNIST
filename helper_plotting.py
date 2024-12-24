import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
    
    
def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs+1), 
             train_acc, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

def plot_input_images(data_loader, unnormalizer = None, figsize= (25, 2.5), n_images = 15):
    fig, axes = plt.subplots(1, n_images, figsize = figsize)
    for batch_idx, (features, _) in enumerate(data_loader):
        orig_images = features[:n_images] 
        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        break
    
    for i in range(n_images):
        for ax, img in zip(axes, orig_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')
    
        
def plot_input_images_with_vae(data_loader, vae_model, device, unnormalizer=None, figsize=(25, 5), n_images=15, noise_std=0.05):
    """
    Plots original images and their VAE reconstructions side by side.

    Parameters:
    - data_loader: DataLoader containing the images to plot.
    - vae_model: The trained VAE model for reconstructing images.
    - unnormalizer: Function to unnormalize images (optional).
    - figsize: Tuple specifying the size of the plot.
    - n_images: Number of images to display.
    - noise_std: Standard deviation of the Gaussian noise to add to images.
    """
    fig, axes = plt.subplots(2, n_images, figsize=figsize)

    # Ensure axes are iterable even for single rows
    if n_images == 1:
        axes = [axes]

    # Extract a batch of images
    for batch_idx, (features, _) in enumerate(data_loader):
        orig_images = features[:n_images]
        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        break

    # Process each image
    for i, (ax_orig, ax_vae, img) in enumerate(zip(axes[0], axes[1], orig_images)):
        # Prepare the original image
        curr_img = img.detach().to(device)

        # Add Gaussian noise
        noise = torch.randn_like(curr_img) * noise_std
        noisy_img = curr_img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)

        # Pass the image through the VAE
        with torch.no_grad():
            _, _, _, vae_img = vae_model(noisy_img.unsqueeze(0))  # Add batch dimension
            vae_img = vae_img.squeeze(0).to(torch.device('cpu'))

        noisy_img = noisy_img.detach().to(torch.device('cpu'))
        # Unnormalize if needed
        if unnormalizer is not None:
            noisy_img = unnormalizer(noisy_img)
            vae_img = unnormalizer(vae_img)

        # Handle color channels for original image
        if color_channels > 1:
            noisy_img = np.transpose(noisy_img.numpy(), (1, 2, 0))
            ax_orig.imshow(noisy_img)
        else:
            ax_orig.imshow(noisy_img.view((image_height, image_width)).numpy(), cmap='binary')

        # Handle color channels for VAE image
        if color_channels > 1:
            vae_img = np.transpose(vae_img.numpy(), (1, 2, 0))
            ax_vae.imshow(vae_img)
        else:
            ax_vae.imshow(vae_img.view((image_height, image_width)).numpy(), cmap='binary')

        # Turn off axes for clean visualization
        ax_orig.axis('off')
        ax_vae.axis('off')

    # Add titles
    for ax in axes[0]:
        ax.set_title("Original + Noise")
    for ax in axes[1]:
        ax.set_title("Reconstruction")

    plt.tight_layout()
    
def plot_generated_images(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 2.5), n_images=15, modeltype='autoencoder'):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (features, _) in enumerate(data_loader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            if modeltype == 'autoencoder':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                encoded, z_mean, z_log_var, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')
                
                
def plot_latent_space_with_labels(num_classes, data_loader, encoding_fn, device):
    d = {i:[] for i in range(num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            
            embedding = encoding_fn(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            color=colors[i][1],
            label=f'{i}',
            alpha=0.5)

    plt.legend()
    
    
def plot_images_sampled_from_vae(model, device, latent_size, unnormalizer=None, num_images=10):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size).to(device)
        new_images = model.decoder(rand_features)
        color_channels = new_images.shape[1]
        image_height = new_images.shape[2]
        image_width = new_images.shape[3]

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary') 
                
                
        



