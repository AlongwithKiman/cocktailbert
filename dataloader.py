import os
import torch
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pytorch_lightning as pl
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import random_split, Dataset, DataLoader
device = torch.device("cuda:0")


class NSMCDataset(Dataset):

    def __init__(self, file_path, max_seq_len, num_categories):
        self.data = pd.read_csv(file_path)
        self.max_seq_len = max_seq_len
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.num_categories = num_categories

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        doc = data['sentence']
        features = self.tokenizer.encode_plus(str(doc),
                                              add_special_tokens=True,
                                              max_length=self.max_seq_len,
                                              pad_to_max_length='longest',
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                             )
        input_ids = features['input_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)
        token_type_ids = features['token_type_ids'].squeeze(0)
        label = [torch.tensor(data[f'label{i}']) for i in range(self.num_categories)]
        #label = torch.tensor(data['label'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label
        }


class NSMCDataModule(pl.LightningDataModule):
    def __init__(self, data_path, mode, valid_size, max_seq_len, batch_size, num_categories):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.valid_size = valid_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_categories = num_categories

    def prepare_data(self):
        # Assuming data preparation steps here (e.g., downloading, loading, and preprocessing)
        pass

    def setup(self, stage=None):
        # Load data and split into train and validation sets
        dataset = NSMCDataset(self.data_path, self.max_seq_len,self.num_categories)
        total_samples = len(dataset)
        valid_samples = int(total_samples * self.valid_size)
        train_samples = total_samples - valid_samples
        self.train_dataset, self.valid_dataset = random_split(dataset, [train_samples, valid_samples])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Assuming test dataset is available, implement if needed
        pass