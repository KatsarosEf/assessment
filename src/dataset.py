import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

class MRIDataset(data.Dataset):
    """
    Constructs the MRIDataset.
    """
    def __init__(self, root_dir, task, plane, plane_f, transform=None):
        """ Initializes the Pytorch dataset class

        Parameters
        ----------
        root_dir: str, directory where data resides
        task: str, type of classification task to perform
        plane: str, type of input mri plane to be used, i.e. axial
        plane_f: str (Optional), determines whether a second plane should be employed, i.e. for fusion
        transform: str (Optional), transformations to be applied on the mri input data
        """
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.plane_f = plane_f

        self.folder_path = self.root_dir + '/{0}/'.format(plane)
        self.records = pd.read_csv(self.root_dir + '-{0}.csv'.format(task), header=None, names=['id', 'label'])
        self.records['id'] = self.records['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))

        self.paths = [self.folder_path + filename + '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()
        self.transform = transform

        if plane_f:
            self.folder_path_f = self.root_dir + '/{0}/'.format(self.plane_f )
            self.paths_f = [self.folder_path_f + filename + '.npy' for filename in self.records['id'].tolist()]

        # determine the weights to scale the loss for class imbalance
        pos = np.sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([1, neg / pos])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """ Used by the dataloader to form batches and parallelize fetching.

        Parameters
        ----------
        index: int, indicates a sample of the dataset

        Returns
        -------
        array: torch.tensor or tuple of two torch.tensors, mri from one or two planes respectively
        label: torch.tensor, classification labels for the slices at hand
        weights: torch.tensor, global weights computed once at initialization to scale the loss
        """
        # source and serve the first mri plane
        array = np.load(self.paths[index])
        label = self.labels[index]
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])

        if self.transform:
            array = self.transform(array)
        else:
            array = torch.FloatTensor(np.stack((array,)*3, axis=1))

        # if fusion, then source and serve the second mri plane too
        if self.plane_f:
            array_f = np.load(self.paths_f[index])
            if self.transform:
                array_f = self.transform(array_f)
            else:
                array_f = torch.FloatTensor(np.stack((array_f,) * 3, axis=1))

            return (array, array_f), label, self.weights

        else:
            return array, label, self.weights
