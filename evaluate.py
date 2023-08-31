import os
import argparse

import torch
from src.dataset import MRIDataset
from src.models import FusedAlexNet, AlexNet
from sklearn import metrics


def test(args):
    """ Evaluates a trained model under one of the two configurations i.e. "axial" or "axial&sagittal" as per the readme
        and prints the area under the curve, the f1-score and the accuracy of on the "abnormal" classification task.
    Args:
        args: argsparse.Namespace

    """

    data = MRIDataset(os.path.join(args.data_path, 'valid'), 'abnormal', args.plane, args.fusion, transform=None)
    loader = torch.utils.data.DataLoader(data, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download and load state_dict
    if args.plane=='axial' and args.fusion==None:
        model = AlexNet().to(device)
        model_weights_zip = './models/axial/single.pth'
        model_weights = torch.load(model_weights_zip, map_location=device)['state']
        model.load_state_dict(model_weights)

    elif args.plane=='axial' and args.fusion=='sagittal':
        model = FusedAlexNet().to(device)
        model_weights_zip = './models/axial&sagittal/fused.pth'
        model_weights = torch.load(model_weights_zip, map_location=device)['state']
        model.load_state_dict(model_weights)

    else:
        print("No such trained model, exiting...")
        exit()


    model.eval()
    y_true, y_prob, y_pred = [], [], []

    with torch.no_grad():
        for idx, (image, label, weight) in enumerate(loader):

            # mount data on gpu
            image = [x.to(device) for x in image]
            label, weight = [x.to(device) for x in [label, weight]]

            # make predictions
            scores = model.forward(image)
            probs = torch.sigmoid(scores)
            _, preds = torch.max(probs, 1)

            y_true.append(int(label.flatten()[-1]))
            y_prob.append(probs.flatten()[-1].item())
            y_pred.append(preds.item())
            print('[IDX: {}/{}]'.format( idx+1, len(loader)))

        auc = metrics.roc_auc_score(y_true, y_prob)
        acc = metrics.accuracy_score(y_true, y_pred)
        f1  = metrics.f1_score(y_true, y_pred)
        print('[TEST] Accuracy: {:.4f} AUC: {:.4f} F1: {:.4f}\t'.format(acc, auc, f1))




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'], default='axial', help='MRI plane to use as input')
    parser.add_argument('-f', '--fusion', type=str,
                        choices=['sagittal', 'coronal', 'axial'], help='Complementary input MRI plane')
    parser.add_argument('--data_path', type=str, default='./data/', help='Data directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    test(args)