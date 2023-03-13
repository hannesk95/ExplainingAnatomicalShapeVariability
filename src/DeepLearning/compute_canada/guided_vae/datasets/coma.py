import os
import os.path as osp
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from utils.read import read_mesh
import random
from tqdm import tqdm


class CoMA(InMemoryDataset):
    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

    def __init__(self,
                 root,
                 data_split,
                 split='interpolation',
                 test_exp='bareteeth',
                 transform=None,
                 pre_transform=None):
        self.split = split
        # self.test_exp = test_exp
        if not osp.exists(osp.join(root, 'processed', self.split)):
            os.makedirs(osp.join(root, 'processed', self.split))
        # if self.split == 'extrapolation':
        #     if self.test_exp not in self.categories:
        #         raise RuntimeError(
        #             'Expected expressions in {}, but found {}'.format(
        #                 self.categories, self.test_exp))
        #     if not osp.exists(
        #             osp.join(root, 'processed', self.split, self.test_exp)):
        #         os.makedirs(
        #             osp.join(root, 'processed', self.split, self.test_exp))
        super().__init__(root, transform, pre_transform)
        
        if data_split == "train":
            path = self.processed_paths[0]
        elif data_split == "val":
            path = self.processed_paths[1]
        elif data_split == "test":
            path = self.processed_paths[2]
        else:
            raise RuntimeError('Expected data split to be in ["train", "val", "test"]')            
        
        #path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'COMA_data.zip'

    @property
    def processed_file_names(self):
        # if self.split == 'extrapolation':
        #     return [
        #         osp.join(self.split, self.test_exp, 'training.pt'),
        #         osp.join(self.split, self.test_exp, 'test.pt')
        #     ]
        if self.split == 'interpolation':
            return [
                osp.join(self.split, 'training.pt'),
                osp.join(self.split, 'val.pt'),
                osp.join(self.split, 'test.pt')                
            ]
        else:
            raise RuntimeError(
                ('Expected the split of interpolation or extrapolation, but'
                 ' found {}').format(self.split))

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download COMA_data.zip from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        print('Processing...')

        labels = torch.load(f"./data/CoMA/raw/hippocampus/labels.pt")

        #X_train, X_test, y_train, y_test = train_test_split(list(labels.keys()), list(labels.values()), stratify=list(labels.values()), test_size=0.2, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(list(labels.keys()), list(labels.values()), test_size=0.2, random_state=28)
        #random.seed(3783)
        #indices = random.sample(range(0, 101), 51)
        #X_val = [X_test[index] for index in indices]
        #X_test = [j for i, j in enumerate(X_test) if i not in indices]
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=28)

        #X_val = X_test[:51]
        #X_test_new = X_test[51:]

        fps = glob(osp.join(self.raw_dir, 'hippocampus/*.ply'))
        if len(fps) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            fps = glob(osp.join(self.raw_dir, '*/*/*.ply'))

        train_data_list, val_data_list, test_data_list  = [], [], []
        train_val_test_files = {"train": X_train, "val": X_val, "test": X_test}
        for idx, fp in enumerate(tqdm(fps)):
            data = read_mesh(fp)
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            subject = fp.split("/")[-1].split(".")[0]
            if self.split == 'interpolation':
                #if (idx % 100) < 10:
                if subject in X_test:
                    test_data_list.append(data)
                    #train_val_test_files["test"].append(fp.split("/")[-1])
                elif subject in X_val:
                    val_data_list.append(data)
                elif subject in X_train:
                    train_data_list.append(data)
                    #train_val_test_files["train"].append(fp.split("/")[-1])
                else:
                    raise RuntimeError('ERROR...')

            # elif self.split == 'extrapolation':
            #     if fp.split('/')[-2] == self.test_exp:
            #         test_data_list.append(data)
            #     else:
            #         train_data_list.append(data)
            else:
                raise RuntimeError((
                    'Expected the split of interpolation or extrapolation, but'
                    ' found {}').format(self.split))

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(train_val_test_files, os.path.join(self.root, "processed/train_val_test_files.pt"))
        torch.save(train_data_list, os.path.join(self.root, "processed/train_meshes.pt"))
        torch.save(val_data_list, os.path.join(self.root, "processed/val_meshes.pt"))
        torch.save(test_data_list, os.path.join(self.root, "processed/test_meshes.pt"))
