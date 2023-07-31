import os
import numpy as np
import lmdb
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LmdbDataset(Dataset):

    def __init__(self, size=128, lmdb_dir='data/flower_lmdb'):
        super(LmdbDataset, self).__init__()
        self.dataset = []
        env = lmdb.open(lmdb_dir)
        self.lmdb_dataset = env.begin(write=False)
        self.size = int(self.lmdb_dataset.get('size'.encode()).decode())
        print(f'==> LmdbDataset: Got total {self.size} images.')

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(size + int(0.15 * size)),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image_bin = self.lmdb_dataset.get(f'{index}'.encode())
        # 将二进制文件转为十进制文件（一维数组）
        image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        # 将数据转换(解码)成图像格式
        # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
        image = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.train_transform(image_pil)
        return image_tensor


if __name__ == '__main__':
    flower_dataset = LmdbDataset(128, './data/flower_lmdb')
    a = flower_dataset[0]
    print(a.max(), a.min())
