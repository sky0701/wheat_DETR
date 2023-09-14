import pandas as pd
import numpy as np
import re
import cv2

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt

def expand_bbox(x):
    """
    [0-9]+: 숫자가 하나이상 
    [.]?: 소수점에 사용될 .이 0 or 1번 나타나야 함
    [0-9]*: 소수점 다음에 오는 숫자가 0번이상
    """
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    r = r.astype(np.float64)
    r = [r[0], r[1], r[0]+r[2], r[1] + r[3]]
    
    # 데이터 없으면 -1 할당
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

# data path 
DIR_INPUT = '/data/'
DIR_TRAIN = f'{DIR_INPUT}train/'
DIR_TEST = f'{DIR_INPUT}test/'

train_df = pd.read_csv(DIR_INPUT + 'train.csv')

# bbox [xmin, ymin, width, height] 형태 -> [xmin, ymin, xmax, ymax]로 변경
train_df['xmin'] = -1
train_df['ymin'] = -1
train_df['xmax'] = -1
train_df['ymax'] = -1

train_df[['xmin', 'ymin', 'xmax', 'ymax']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

# 서로 다른 이미지 개수
image_ids = train_df['image_id'].unique()

# 20퍼 validation으로 사용
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

# 같은 이미지 데이터, bbox가 다른 데이터가 있음을 인지
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]


class WheatDataset(Dataset):
    def __init__(self, df, image_dir, transforms = None):
        super().__init__()

        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index:int):
        image_id = self.image_ids[index]

        # 해당 image_id에 대한 모든 데이터 저장
        records = self.df[self.df['image_id'] == image_id]
        # print(records)
        # print(records['xmin'])

        # 이미지 불러오기 및 스케일 조정, 입력 데이터 [0, 1]로 맞춰줌으로써 학습 안정화
        # 입력 데이터의 스케일이 크면 gradient 업데이트가 불안정
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        target = {}
        target['boxes'] = torch.tensor(np.array([records['xmin'].values, records['xmax'].values, records['ymin'].values, records['ymax'].values]))
        target['labels'] = torch.ones((records.shape[0],), dtype=torch.int64)
        target['image_id'] = torch.tensor([index])
        target['area'] = (records['xmax'] - records['xmin']) * (records['ymax'] - records['ymin'])

        return image, target['labels'], target['boxes'], image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


if __name__ == "__main__":
    train_dataset = WheatDataset(train_df, DIR_TRAIN, None)
    train_data_loader = DataLoader(train_dataset, batch_size = 1, shuffle = False)

    for image, labels, boxes, image_id in train_data_loader:
        boxes = boxes.permute(0, 2, 1).squeeze().numpy().astype(np.int32)
        image = image[0].permute(1, 0, 2).numpy()
        image = np.ascontiguousarray(image, dtype=np.uint8)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        for box in boxes:
            print(box)
            print(image.shape)
            cv2.rectangle(image,
                        (box[0], box[2]),
                        (box[1], box[3]),
                        (220, 0, 0), 3)
            
        ax.set_axis_off()
        ax.imshow(image)

        # print(boxes)
        break

