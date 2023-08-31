import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):

    def __init__(self):
        """
        Initializes an AlexNet-based model for binary classification.
        """
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.tensor, the input plane mri slices

        Returns
        -------
        output: torch.tensor, binary output classification scores
        """
        x = torch.squeeze(x[0], dim=0)
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output



class FusedAlexNet(nn.Module):
    def __init__(self):
        """
        Initializes a two-stream AlexNet-based model for binary classification.
        """
        super().__init__()
        self.pretrained_model_a = models.alexnet(pretrained=True)
        self.pooling_layer_a = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model_b = models.alexnet(pretrained=True)
        self.pooling_layer_b = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        """ Takes one tuple with the 2 input planes and yields the classification scores.

        Parameters
        ----------
        x: tuple, contains two input plane MRI slices, each of which is a torch.tensor

        Returns
        -------
        output: torch.tensor, binary output classification scores
        """

        # unpack the tuple
        x1, x2 = x

        # extract feats from first mri plane i.e. axial
        x1 = torch.squeeze(x1, dim=0)
        features_a = self.pretrained_model_a.features(x1)
        pooled_features_a = self.pooling_layer_a(features_a)
        pooled_features_a = pooled_features_a.view(pooled_features_a.size(0), -1)
        flattened_features_a = torch.max(pooled_features_a, 0, keepdim=True)[0]

        # extract feats from second mri input plane, i.e. sagittal
        x2 = torch.squeeze(x2, dim=0)
        features_b = self.pretrained_model_b.features(x2)
        pooled_features_b = self.pooling_layer_b(features_b)
        pooled_features_b = pooled_features_b.view(pooled_features_b.size(0), -1)
        flattened_features_b = torch.max(pooled_features_b, 0, keepdim=True)[0]

        # concatenate and predict the output class
        feats = torch.cat([flattened_features_a, flattened_features_b], 1)
        output = self.classifier(feats)

        return output
