import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import PIL

class Atari(Dataset):
    def __init__(self, root, mode, gamelist=None):
        assert mode in ['train', 'validation', 'test'], f'Invalid dataset mode "{mode}"'
        
        # self.image_path = os.checkpointdir.join(root, f'{key_word}')
        self.image_path = root
        
        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root) \
                    if gamelist is None or fn in gamelist \
                    for img in os.listdir(os.path.join(root, fn, mode))]
        self.image_fn = image_fn
        self.image_fn.sort()
    
    def __getitem__(self, index):
        fn = self.image_fn[index]
        
        pil_img = Image.open(os.path.join(self.image_path, fn)).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        
        image = np.array(pil_img)
        
        image_t = torch.from_numpy(image / 255).permute(2, 0, 1).float()
        
        return image_t
    
    def __len__(self):
        return len(self.image_fn)
