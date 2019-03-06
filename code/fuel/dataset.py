import os
import random
import glob
import pickle
import logging

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.vocab import Vocabulary


class ImgCapDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        super(ImgCapDataset, self).__init__()
        self.logger = logging.getLogger('trainer')

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(self.data_dir, 'images', '{}2014'.format(self.split))
        self.filenames = glob.glob(os.path.join(self.images_dir, '*.jpg'))

        self.vocab, self.captions = self.load_captions(os.path.join(self.data_dir, 'captions'))


    def filename2id(self, filename):
        return str(int(filename.split('/')[-1].split('.')[0].split('_')[-1]))


    def load_captions(self, path):
        self.logger.info('Loading {} Raw Captions...'.format(self.split))
        with open(os.path.join(path, '{}_cap.pickle'.format(self.split)), 'rb') as f:
            caps = pickle.load(f)

        self.logger.info('Loading Vocabulary...')
        with open(os.path.join(path, 'vocab.pickle'), 'rb') as f:
            vocab = pickle.load(f)

        self.logger.info('Vocabulary Size: {}.'.format(len(vocab)))
        
        return vocab, caps

    def get_image(self, image_id):
        img_path = os.path.join(self.images_dir, 'COCO_{}2014_{:0>12}.jpg'.format(self.split, image_id))
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img
    
    def get_caption(self, image_id):
        caps = self.captions[image_id]
        
        cap_word = random.choice(caps)
        cap_idx = [self.vocab(token) for token in cap_word]

        # print(len(cap_word), cap_idx)
        return cap_idx

                
    def __getitem__(self, index):
        image_id = self.filename2id(self.filenames[index])

        image = self.get_image(image_id)
        image = torch.Tensor(image)

        cap_idx = self.get_caption(image_id)
        cap_idx = torch.Tensor(cap_idx)
        
        return image, cap_idx
        
    
    def __len__(self):
        return len(self.filenames)
    

    def name(self):
        return 'Image and Caption Dataset'


def train_transform(crop_size):
    transform_list = [
        transforms.Resize(crop_size, interpolation=Image.BILINEAR),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[-1]), reverse=True)
    images, cap_idxs = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    cap_lens = [len(cap) for cap in cap_idxs]
    padded_caps = torch.zeros(len(cap_idxs), max(cap_lens)).long()
    
    for i, cap in enumerate(cap_idxs):
        end = cap_lens[i]
        padded_caps[i, :end] = cap[:end]
    
    return images, padded_caps, cap_lens