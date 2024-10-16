import torch
import torchvision

from ops.dataset import CombinedDataset

from assembly101.action_recognition.dataset_config import return_dataset
from assembly101.action_recognition.transforms import Stack, ToTorchFormatTensor, GroupMultiScaleCrop, GroupRandomHorizontalFlip, normalize

args = {}
num_class, args.train_list, args.val_list, args.root_path, prefix = (
    return_dataset(args.dataset, args.modality)
)

dataset = CombinedDataset()

tsn_dataset = {
    "root_path": args.root_path,
    "list_file": args.train_list,
    "num_segments": args.num_segments,
    "new_length": 1,
    "modality": args.modality,
    "image_tmpl": prefix,
    "transform": torchvision.transforms.Compose(
        [
            torchvision.transforms.Compose(
                    [
                        GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66]),
                        GroupRandomHorizontalFlip(is_flow=False),
                    ]
                ),
            Stack(roll=(args.arch in ["BNInception", "InceptionV3"])),
            ToTorchFormatTensor(
                div=(args.arch not in ["BNInception", "InceptionV3"])
            ),
            normalize,
        ]
    ),
    "dense_sample": args.dense_sample,
}