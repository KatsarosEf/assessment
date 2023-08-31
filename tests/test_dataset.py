import unittest
import torch
from src.dataset import MRIDataset

class Test_MRIDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        data = MRIDataset('./data/valid', 'abnormal', 'axial', 'sagittal', transform=None)
        loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

        cls.mri = next(iter(loader))

    def test_mri_tensor_dimensions(self):
        image_tensor_shape = Test_MRIDataset.mri[0][0].shape
        self.assertEqual(image_tensor_shape[0], 1)
        self.assertEqual(image_tensor_shape[2], 3)
        self.assertEqual(image_tensor_shape[3], 256)
        self.assertEqual(image_tensor_shape[4], 256)

    def test_label_tensor_dimensions(self):
        label_tensor_shape = Test_MRIDataset.mri[1].shape
        self.assertEqual(label_tensor_shape[0], 1)
        self.assertEqual(label_tensor_shape[1], 1)
        self.assertEqual(label_tensor_shape[2], 2)

    def test_weight_tensor_dimensions(self):
        weight_tensor_shape = Test_MRIDataset.mri[2].shape
        self.assertEqual(weight_tensor_shape[0], 1)
        self.assertEqual(weight_tensor_shape[1], 2)


