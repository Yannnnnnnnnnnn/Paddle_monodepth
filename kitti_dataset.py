import os
import cv2
import numpy as np

import paddle
from paddle.io import Dataset

class MonodepthDataset(Dataset):

    """monodepth dataloader"""

    def __init__(self, data_path, list_path, mode):

        self.data_path = data_path
        self.mode = mode

        # read file list
        with open(list_path, 'r') as f:
            self.lists = f.readlines()

        self.left_images  = []
        self.right_images = []
        for list in self.lists:
            
            left_image,right_image = list.strip().split(' ')

            self.left_images.append(os.path.join(data_path,left_image).replace('jpg','png'))
            self.right_images.append(os.path.join(data_path,right_image).replace('jpg','png'))

    def __len__(self):
        return len(self.left_images)

    def augment_image_pair(self, left_image, right_image):

        # shape
        H,W,C = left_image.shape

        # randomly shift gamma
        random_gamma = paddle.uniform(shape=[1],min=0.8, max=1.2)
        left_image_aug  = left_image  ** random_gamma[0]
        right_image_aug = right_image ** random_gamma[0]

        # randomly shift brightness
        random_brightness = paddle.uniform([1], min=0.5, max=2.0)
        left_image_aug  =  left_image_aug * random_brightness[0]
        right_image_aug = right_image_aug * random_brightness[0]

        # randomly shift color
        random_colors = paddle.uniform(shape=[3], min=0.8, max=1.2)
        white = paddle.ones((H,W,1))
        color_image = paddle.concat([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug  = paddle.clip(left_image_aug,  0, 1)
        right_image_aug = paddle.clip(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        
        image = cv2.imread(image_path)
        image = np.asarray(image, dtype=np.float32)/255.0
        image = paddle.to_tensor(image)
        return image

    def __getitem__(self, idx):

        left_image = self.read_image(self.left_images[idx])
        right_image = self.read_image(self.right_images[idx])

        if self.mode == 'train':
            left_image,right_image = self.augment_image_pair(left_image,right_image)

        left_image = left_image.transpose((2,0,1))
        right_image = right_image.transpose((2,0,1))

        return left_image,right_image

if __name__ == '__main__':

    kitti_dataset = MonodepthDataset(data_path='C:/Users/MSI-1/Desktop/KITTI/',list_path='./utils/filenames/kitti_test.txt',mode='train')

    left_image,right_image = kitti_dataset[0]
    print('left',left_image.shape)

    import matplotlib.pyplot as plt
    plt.imshow(left_image.transpose((1,2,0)))
    plt.show()
