from data.dataset import WMT14ENDeDataset

from data.dataset import WMT14ENDeDataset

train_dataset = WMT14ENDeDataset(data_dir='/data/raw', split='train')
valid_dataset = WMT14ENDeDataset(data_dir='/data/raw', split='valid')
test_dataset = WMT14ENDeDataset(data_dir='/data/raw', split='test')

