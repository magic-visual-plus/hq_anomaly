

from torch.utils.data import Dataset
import os
import cv2


class ImageSingleFolder(Dataset):
    def __init__(self, folder, transform=None):
        super().__init__()
        self.folder = folder
        self.transforms = transform
        self.filenames = os.listdir(self.folder)
        self.filenames = [f for f in self.filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        pass
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.filenames[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(img)
            pass
        
        return img