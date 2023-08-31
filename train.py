import os
import argparse
import wandb

from src.dataset import MRIDataset
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor
from torchvision import transforms

import torch
from src.utils import tmp_function, train_epoch, validate_epoch, model_save
import torch.optim as optim
from src.models import FusedAlexNet, AlexNet



def run(args):
    """
    Loads dataset, dataloaders, trains, validates and saves weights.

    Parameters
    ----------
    args: argsparse.Namespace

    """

    # make transformations, data, loaders and configure device
    transformations = {'train': transforms.Compose([ToTensor(),
                                                    RandomRotate(25),
                                                    RandomTranslate([0.11, 0.11]),
                                                    RandomFlip(),
                                                    transforms.Lambda(tmp_function)
                                                    ]),
                       'valid': None}

    data = {x: MRIDataset(os.path.join(args.data_path, x), args.task, args.plane,
                          args.fusion, transform=transformations[x]) for x in ['train', 'valid']}

    loader = {x: torch.utils.data.DataLoader(data[x], batch_size=1, shuffle=True,
                          num_workers=1, drop_last=False) for x in ['train', 'valid']}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize model, optim and scheduler
    model = FusedAlexNet().to(device) if args.fusion else AlexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # initialize wandb logger
    wandb.init(project='mri-cv', entity='ekatsaros', mode='disabled')
    wandb.run.name = args.out.split('/')[-1]
    wandb.watch(model)

    # make path if not there
    if not os.path.exists(os.path.join('./runs', args.out)):
        os.makedirs(os.path.join('./runs', args.out))

    # train, val and model save
    for epoch in range(1, args.epochs+1):

        train_epoch(args, loader['train'], model, optimizer, scheduler, epoch, device)

        validate_epoch(args, loader['valid'], model, epoch, device)

        model_save(args, model, optimizer, scheduler, epoch)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'], default='abnormal', help='Malfunction to diagnose')
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'], default='axial', help='MRI plane to use as input')
    parser.add_argument('-f', '--fusion', type=str,
                        choices=['sagittal', 'coronal', 'axial'], help='Complementary input MRI plane')
    parser.add_argument('--data_path', type=str, default='./data/', help='Data directory')
    #parser.add_argument('--data_path', type=str, default='../raid/data/')
    parser.add_argument('--out', type=str, default='trial', help='Subdir in ./runs to store experiment weights')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    args = parser.parse_args()


    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)