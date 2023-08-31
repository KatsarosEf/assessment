import os
import torch
import torch.nn as nn
import wandb
from sklearn import metrics

def train_epoch(args, loader, model, optimizer, scheduler, epoch, device):
    """ Trains model for one epoch and logs training stats.

    Parameters
    ----------
    args: argsparse.Namespace, arguments provided from terminal
    loader: torch.utils.data.DataLoader, pytorch dataloader
    model: torchvision.models.AlexNet, pytorch model
    optimizer: torch.optim, pytorch optimizer
    scheduler: torch.optim.lr_scheduler, pytorch learning rate scheduler
    epoch: int, current training epoch
    device: torch.device, device to use, i.e. cuda or cpu
    """

    # switch intro training mode and initialize statistics
    model.train()
    running_corrects, running_loss = 0, 0.0
    # initialize fictitious predictions to circumvent AUC not-defined-with-one-class
    y_true, y_pred = [0, 1], [0.5, 0.5]

    for idx, (image, label, weight) in enumerate(loader):

        # zero gradients and mount data on cuda
        optimizer.zero_grad()
        image = [x.to(device) for x in image]
        label, weight = [x.to(device) for x in [label, weight]]

        # make predictions, compute loss, compute gradients, update weights
        scores = model.forward(image)
        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(scores, label[0])
        loss.backward()
        optimizer.step()

        # extract probabilities and measure metrics
        probs = torch.sigmoid(scores)
        _, preds = torch.max(probs, 1)
        _, true = torch.max(label, 2)
        running_corrects += preds==true
        running_acc = running_corrects.item() / (idx+1)
        running_loss += loss.item() / (idx+1)

        y_true.append(int(label.flatten()[-1]))
        y_pred.append(probs.flatten()[-1].item())
        running_auc = metrics.roc_auc_score(y_true, y_pred)

        print('[TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f} AUC: {:.4f}\t'.format(
            epoch, args.epochs, idx + 1, len(loader), loss.item(), running_acc, running_auc))

    # Log i.e. loss function and epoch metrics
    train_metrics = {'train_loss': running_loss,
                     'train_accuracy': running_acc,
                     'train_auc': running_auc,
                     'epoch': epoch}
    wandb.log(train_metrics)
    scheduler.step()



def validate_epoch(args, loader, model, epoch, device):
    """ Validates the model after each epoch and logs the metrics.

    Parameters
    ----------
    args: argsparse.Namespace, arguments provided by terminal
    loader: torch.utils.data.DataLoader, pytorch dataloader
    model: torchvision.models.AlexNet, pytorch model
    epoch: int, current epoch
    device: torch.device, device to validate on i.e. cpu or cuda
    """

    # switch intro eval mode
    model.eval()
    running_corrects, running_loss = 0, 0.0
    y_true, y_pred = [0, 1], [0.5, 0.5]

    with torch.no_grad():
        for idx, (image, label, weight) in enumerate(loader):

            # mount data on gpu
            image = [x.to(device) for x in image]
            label, weight = [x.to(device) for x in [label, weight]]

            # make predictions
            scores = model.forward(image)
            loss = nn.BCEWithLogitsLoss(weight=weight)(scores, label[0])

            # extract probabilities and measure metrics
            probs = torch.sigmoid(scores)
            _, preds = torch.max(probs, 1)
            _, true = torch.max(label, 2)
            running_corrects += preds == true
            running_acc = running_corrects.item() / (idx + 1)
            running_loss += loss.item() / (idx + 1)

            y_true.append(int(label.flatten()[-1]))
            y_pred.append(probs.flatten()[-1].item())
            running_auc = metrics.roc_auc_score(y_true, y_pred)

            print('[VALIDATION] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f} AUC: {:.4f}\t'.format(
                epoch, args.epochs, idx + 1, len(loader), loss.item(), running_acc, running_auc))

        # log i.e. loss function and epoch metrics
        val_metrics = {'val_loss': running_loss,
                       'val_accuracy': running_acc,
                       'val_auc': running_auc,
                       'epoch': epoch}
        wandb.log(val_metrics)


def model_save(args, model, optimizer, scheduler, epoch):
    """ Saves model, optimizer and scheduler history at some specific epoch.

    Parameters
    ----------
    args: argsparse.Namespace, arguments
    model: torchvision.models.AlexNet, an instantiation of an AlexNet pytorch model
    optimizer:
    scheduler:
    epoch: int, current epoch in training loop
    """
    save_dict = {
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    torch.save(save_dict, os.path.join('./runs', args.out, 'ckpt_{}.pth'.format(epoch)))


def tmp_function(x):
    # multi-processing is not natively supported in windows
    # i.e. lambda breaks throwing pickling errors
    # this function circumvents the issue for the specific transform
    return x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
