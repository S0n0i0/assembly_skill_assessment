from torch.utils.data import Dataset
import json
from enum import Enum

from assembly101.action_anticipation.dataset import SequenceDataset
from assembly101.action_recognition.dataset import TSNDataSet
from assembly101.temporal_action_segmentation.dataset import AugmentDataset

from utils.classes import PathSource
from utils.enums import PathType, LogCode, DisplayLevel
from utils.constants import log_manager


class DatasetOpCode(Enum):
    """Enum class representing dataset Operation Codes in logs."""

    _LOAD = 0


class CombinedDataset(Dataset):
    def __init__(
        self,
        sequence_dataset_args,
        tsn_dataset_args,
        augment_dataset_args,
        pose_source: PathSource,
        gaze_source: PathSource | None = None,
        object_source: PathSource | None = None,
        gaze_target: PathSource | None = None,
    ):
        try:
            if gaze_target is not None:
                if gaze_target.type != PathType.FILE:
                    raise ValueError("Gaze target must be a file")
                self.gaze_target: dict | None = json.load(open(gaze_target.path))
            else:
                self.gaze_target: dict | None = None

            if (
                self.gaze_target is None
                and gaze_source is not None
                and object_source is not None
            ):
                if gaze_source.type != PathType.FILE:
                    raise ValueError("Gaze source must be a file")
                self.gaze_source: dict | None = json.load(open(gaze_source.path))

                if object_source.type != PathType.FILE:
                    raise ValueError("Object source must be a file")
                self.object_source: dict | None = json.load(open(object_source.path))

            if (
                self.gaze_target is None
                and self.gaze_source is None
                and self.object_source is None
            ):
                raise ValueError(
                    "Gaze target or gaze source and object source must be provided"
                )

            if pose_source is not None:
                if pose_source.type != PathType.FILE:
                    raise ValueError("Pose source must be a file")
            self.pose_source: dict | None = json.load(open(pose_source.path))

            self.sequence_dataset = SequenceDataset(**sequence_dataset_args)
            self.tsn_dataset = TSNDataSet(**tsn_dataset_args)
            self.augment_dataset = AugmentDataset(**augment_dataset_args)

        except:
            self.gaze_source = None
            self.object_source = None
            self.gaze_target = None
            self.pose_source = None
            self.sequence_dataset = None
            self.tsn_dataset = None
            self.augment_dataset = None
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                "DATASET_LOAD",
                "Error loading dataset",
                DisplayLevel.HIGH,
            )

    def __len__(self):
        return min(
            len(self.sequence_dataset), len(self.tsn_dataset), len(self.augment_dataset)
        )

    def __getitem__(self, idx):
        video_record, indexes, original_frames, transformed_images, _ = (
            self.tsn_dataset[idx]
        )
        sequence_data = self.sequence_dataset[idx]
        augment_data = self.augment_dataset[idx]

        gaze_target_data = None
        gaze_source_data = None
        object_source_data = None
        if self.gaze_target is not None:
            gaze_target_data = self.gaze_target[video_record.path]
        elif self.gaze_source is not None and self.object_source is not None:
            gaze_source_data = self.gaze_source[video_record.path]
            object_source_data = self.object_source[video_record.path]

        pose_data = None
        if self.pose_source is not None:
            pose_data = self.pose_source[video_record.path]

        # Combine the data as needed
        combined_data = {
            "recognition_data": transformed_images,
            "anticipation_data": {
                "spanning_features": sequence_data["spanning_features"],
                "recent_features": sequence_data["recent_features"],
            },
            "segmentation_data": augment_data[0].permute(0, 2, 1),
            "video_data": {
                "video_record": video_record,
                "indexes": indexes,
                "frames": original_frames,
            },
            "gaze_data": {
                "gaze_target": gaze_target_data,
                "gaze_source": gaze_source_data,
                "object_source": object_source_data,
            },
            "pose_data": pose_data,
        }

        return combined_data
