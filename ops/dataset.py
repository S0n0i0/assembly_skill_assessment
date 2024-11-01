from torch.utils.data import Dataset
import json
from enum import Enum

import assembly101.action_anticipation.dataset as aa_dataset
import assembly101.action_recognition.dataset as ar_dataset
import assembly101.temporal_action_segmentation.dataset as tas_dataset

from utils.classes import PathSource
from utils.enums import (
    PathType,
    LogCode,
    DisplayLevel,
)
from utils.constants import log_manager


class DatasetOpCode(Enum):
    """Enum class representing dataset Operation Codes in logs."""

    _LOAD = 0


class CombinedDataset(Dataset):
    def __init__(
        self,
        # split: SimpleSplits | ComposedSplit | ChallengeSplits,
        # fine_splits_path: PathSource,
        # coarse_annotations_path: PathSource,
        sequence_dataset_args: PathSource | dict,
        tsn_dataset_args: PathSource | dict,
        augment_dataset_args: PathSource | dict,
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
            self.pose_source: dict = json.load(open(pose_source.path))

            if isinstance(sequence_dataset_args, PathSource):
                self.sequence_dataset = None
            else:
                self.sequence_dataset = aa_dataset.SequenceDataset(
                    **sequence_dataset_args
                )
                self.fine_support, self.fine_support_order = self.gen_supports(
                    self.sequence_dataset.video_list,
                    lambda item: item.path.split("/")[0],
                    lambda item: item.path.split("/")[1],
                    lambda item: item.start_frame,
                    lambda item: item.end_frame,
                )
            if isinstance(tsn_dataset_args, PathSource):
                self.tsn_dataset = None
            else:
                self.tsn_dataset = ar_dataset.TSNDataSet(**tsn_dataset_args)
                if isinstance(sequence_dataset_args, PathSource):
                    self.fine_support, self.fine_support_order = self.gen_supports(
                        self.tsn_dataset.video_list,
                        lambda item: item.path.split("/")[0],
                        lambda item: item.path.split("/")[1],
                        lambda item: item.start_frame,
                        lambda item: item.num_frames
                        + item.start_frame
                        - 1,  # 1 = modifier in gen_fine_labels.py
                    )

            if isinstance(augment_dataset_args, PathSource):
                self.augment_dataset = None
            else:
                self.augment_dataset = tas_dataset.AugmentDataset(
                    **augment_dataset_args
                )
                self.coarse_support, self.coarse_order_support = self.gen_supports(
                    self.augment_dataset.data,
                    lambda item: item["video_id"],
                    lambda item: item["view"],
                    lambda item: item["st_frame"],
                    lambda item: item["end_frame"],
                )

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

    def gen_supports(
        self, source: list, sequence_fn, view_fn, start_frame_fn=None, end_frame_fn=None
    ):
        support = {}
        support_order = {}
        for i, item in enumerate(source):
            sequence = sequence_fn(item)
            view = view_fn(item)
            if sequence not in support:
                support[sequence] = {}
            if view not in support[sequence]:
                support[sequence][view] = {}
            start_frame = int(start_frame_fn(item))
            support[sequence][view][start_frame] = {
                "index": i,
                "end_frame": int(end_frame_fn(item)),
            }

            if sequence not in support_order:
                support_order[sequence] = {}
            if view not in support_order[sequence]:
                support_order[sequence][view] = []
            support_order[sequence][view].append(start_frame)

        for sequence in support_order:
            for view in support_order[sequence]:
                support_order[sequence][view].sort()

        return support, support_order
