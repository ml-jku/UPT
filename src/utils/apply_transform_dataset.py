from torch.utils.data import Dataset


class ApplyTransformDataset(Dataset):
    """
    helper dataset to apply a transform in parallel fashion to some data
    applying transforms via the pytorch DataLoader is much faster than applying it via joblib
    (on ImageNet10-M3AE logging embeddings of a ViT-B takes 10:40 with joblib vs 2:40 with pytorch)
    """

    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)
