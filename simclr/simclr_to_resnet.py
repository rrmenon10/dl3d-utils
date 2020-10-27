import torch

from simclr import SimCLR

simclr = SimCLR((32,32),
                model_name='resnet50',
                embedding_size=512,
                temp=0.8,
                transform=True,
                device=torch.device('cuda'))
simclr.load_state_dict(torch.load('./config_simclr_cifar/models/999.pth'))
simclr.to(torch.device('cpu'))
torch.save(simclr.model.state_dict(), 'pretrained_resnet50.pth')
