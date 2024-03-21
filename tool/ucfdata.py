import os
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TestDataset(Dataset):
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size
        # UCF
        self.imgs_path = '/data/wangyh/data4/Datasets/shadow/UCF-shadow/InputImages'
        self.labs_path = '/data/wangyh/data4/Datasets/shadow/UCF-shadow/GroundTruth'
        self.imgs = sorted(os.listdir(self.imgs_path))
        self.labs = sorted(os.listdir(self.labs_path))

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.imgs[index]
        label_path = self.labs[index]
        image = Image.open(os.path.join(self.imgs_path, image_path)).convert('RGB').resize((self.image_size, self.image_size))
        label = Image.open(os.path.join(self.labs_path, label_path)).convert('L').resize((self.image_size, self.image_size))
        # transform
        label = np.array(label, dtype='float32')/255.0
        if len(label.shape) > 2:
            label = label[:,:,0]
        label = np.expand_dims(label, axis=0)
        image_nom = self.trans(image)
        sample = {'O': image_nom, 'B':label, 'image': np.array(image,dtype='float32').transpose(2,0,1)/255}
        return sample, image_path