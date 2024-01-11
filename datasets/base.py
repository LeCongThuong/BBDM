from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import torch
import numpy as np


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name
    
class PrintPaths(Dataset):
    def __init__(self, paths, image_size=(512, 512)):
        self.size = image_size
        self.labels = dict()
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        """Utility function that load an image an convert to torch."""
        # open image using OpenCV (HxWxC)
        img = Image.open(image_path).convert('L')
        # convert image to torch tensor (CxHxW)
        img_t: torch.Tensor = self.transform(img)
        return img_t

    def __getitem__(self, i):
        image = self.preprocess_image(self.labels["file_path_"][i])
        image_label = Path(self.labels["file_path_"][i]).stem
        return image, image_label

class NumpyDepthPaths(Dataset):
    def __init__(self, paths, image_size=(512, 512)):
        self.size = image_size
        self.labels = dict()
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self._length

    def preprocess_depth(self, depth_path):
        """Utility function that load an image an convert to torch."""
        # open image using OpenCV (HxWxC)
        img: np.ndarray = np.load(depth_path)
        # unsqueeze to make it 1xHxW
        img = np.expand_dims(img, axis=0)
        # cast type as np.float32
        img = img.astype(np.float32)
        # convert image to torch tensor (CxHxW)
        img_t: torch.Tensor = torch.from_numpy(img)
        return img_t

    def __getitem__(self, i):
        image = self.preprocess_depth(self.labels["file_path_"][i])
        image_label = Path(self.labels["file_path_"][i]).stem
        return image, image_label


