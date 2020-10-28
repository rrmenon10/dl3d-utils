import os
import sys
import tqdm
import numpy as np

import torch
import torchvision

from simclr import SimCLR
from utils import load_config, plot_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    config_file = sys.argv[1]
    try:
        config = load_config(config_file)
    except:
        print('Please provide a valid config file.')
    
    save_parent_path = '{}_{}'.format(config_file.split(".")[0], config.title)
    save_path = os.path.join(save_parent_path, config.save_path)
    if not os.path.isdir(save_path): os.makedirs(save_path)

    ## Gather dataset
    train_data = torchvision.datasets.CIFAR10(config.data_location,
                                              train=True,
                                              download=True,
                                              transform=torchvision.transforms.ToTensor())
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(train_data,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=4)
    # Define model
    simclr = SimCLR(np.array(train_data[0][0]).shape[1:],
                    model_name=config.simclr.model_name,
                    embedding_size=config.simclr.embedding_size,
                    temp=config.simclr.temp,
                    transform=config.simclr.transforms,
                    device=device)
    # Define optimizer and learning rate schedule
    opt = torch.optim.Adam(simclr.parameters(),
                           lr=config.opt.lr,
                           weight_decay=config.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                           T_max=len(dataloader))
    

    # Start training
    loss_stats = []
    for epoch in tqdm.tqdm(range(config.num_epochs)):
        for (x, _) in tqdm.tqdm(dataloader):
            loss = simclr(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_stats.append(float(loss.data.cpu().numpy()))
        
        if epoch >=10:
            scheduler.step()
        
        print('Plotting after epoch {}/{}'.format(epoch+1, config.num_epochs))
        plot_loss(loss_stats,
				  title=config.title + ' training curve',
				  path=os.path.join(save_parent_path, 'loss.png'))
        
        # saving model after every 100 epochs
        if epoch%100==0:
            PATH = save_path + '{}.pth'.format(epoch)
            torch.save(simclr.state_dict(), PATH)

    PATH = save_path + '{}.pth'.format(epoch)
    torch.save(simclr.state_dict(), PATH)

    np.save(os.path.join(save_parent_path, 'loss.npy'), loss_stats, allow_pickle=True)

if __name__ == '__main__':
    main()
