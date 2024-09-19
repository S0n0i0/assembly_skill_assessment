from torch.utils.data import Dataset
from assembly101.action_anticipation.dataset import SequenceDataset
from assembly101.action_recognition.dataset import TSNDataSet
from assembly101.temporal_action_segmentation.dataset import AugmentDataset


class CombinedDataset(Dataset):
    def __init__(self, sequence_dataset_args, tsn_dataset_args, augment_dataset_args):
        self.sequence_dataset = SequenceDataset(**sequence_dataset_args)
        self.tsn_dataset = TSNDataSet(**tsn_dataset_args)
        self.augment_dataset = AugmentDataset(**augment_dataset_args)

    def __len__(self):
        return min(
            len(self.sequence_dataset), len(self.tsn_dataset), len(self.augment_dataset)
        )

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        tsn_data = self.tsn_dataset[idx]
        augment_data = self.augment_dataset[idx]

        # Combine the data as needed
        combined_data = {
            "sequence_data": sequence_data,
            "tsn_data": tsn_data,
            "augment_data": augment_data,
        }

        return combined_data
