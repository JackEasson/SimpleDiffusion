import os
import lmdb
import cv2
import numpy as np
import random
from tqdm import tqdm


def create_lmdb_dataset(root_dir, save_dir, size_g: float = 0.1):
    B = int(838860800 * size_g)
    env = lmdb.open(save_dir, map_size=B)
    txn = env.begin(write=True)
    folder_name_list = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    idx = 0
    for folder_name in folder_name_list:
        folder_path = os.path.join(root_dir, folder_name)
        print(f'Current process folder: {folder_name}')
        file_name_list = os.listdir(folder_path)
        for file_name in tqdm(file_name_list):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                # 读取图像文件的二进制格式数据
                image_bin = f.read()
                print(idx)
                txn.put(f'{idx}'.encode(), image_bin)
                idx += 1
        #     if idx >= 10:
        #         break
        # break
    txn.put('size'.encode(), f'{idx}'.encode())
    txn.commit()
    env.close()


def read_from_lmdb_dataset(lmdb_dir):
    env = lmdb.open(lmdb_dir)
    with env.begin(write=False) as txn:
        size = int(txn.get('size'.encode()).decode())
        print(size, type(size))
        num = 10
        for idx in range(num):
            cur_idx = random.choice(range(size))
            print(cur_idx)
            image_bin = txn.get(f'{cur_idx}'.encode())
            # 将二进制文件转为十进制文件（一维数组）
            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            # 将数据转换(解码)成图像格式
            # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
            img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
            print(img.shape)
            cv2.imshow('image', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    root_dir = 'E:\\data\\flower_photos'
    save_dir = './flower_lmdb'
    create_lmdb_dataset(root_dir, save_dir, 0.3)
    read_from_lmdb_dataset('flower_lmdb')