from torch.utils.data import Dataset
from enum import Enum
import os
import pickle
import csv
import cv2

import assembly101.action_anticipation.dataset as aa_dataset
import assembly101.action_recognition.dataset as ar_dataset
import assembly101.temporal_action_segmentation.dataset as tas_dataset

from utils.classes import PathSource, SourceMode, VideoRecord
from utils.enums import (
    LogCode,
    DisplayLevel,
    SimpleSplits,
    ComposedSplit,
)
from utils.constants import debug_on, log_manager, view_dict
from base_features_extraction.gaze_tracking import GazeObjectTracking


class DatasetOpCode(Enum):
    """Enum class representing dataset Operation Codes in logs."""

    AA_LOAD = 0
    AA_DUMP_LOAD = 1
    AR_LOAD = 2
    AR_DUMP_LOAD = 3
    TAS_LOAD = 4
    TAS_DUMP_LOAD = 5
    GT_DUMP_LOAD = 6
    GS_DUMP_LOAD = 7
    OS_SUP_DUMP_LOAD = 8
    OS_DUMP_LOAD = 9
    PS_DUMP_LOAD = 10
    DATASET_LOAD = 11


class CombinedDataset(Dataset):
    def __init__(
        self,
        split: SimpleSplits | ComposedSplit,
        annotations_fps: int,
        target_fps: int,
        offsets_paths: dict[str, PathSource],
        skill_splits_path: PathSource,
        image_tmpl: str,
        sequence_dataset_args: PathSource | dict,
        tsn_dataset_args: PathSource | dict,
        augment_dataset_args: PathSource | dict,
        pose_source: PathSource,
        object_source: PathSource | None = None,
        gaze_source: PathSource | None = None,
        gaze_target: PathSource | None = None,
    ):
        global view_dict

        # try:
        if True:
            self.annotations_fps = annotations_fps
            self.target_fps = target_fps
            self.image_tmpl = image_tmpl
            self.offsets: dict[str, dict[str, dict[str, int]]] = {}
            for view_type in view_dict.keys():
                self.offsets[view_type] = {}
                with open(offsets_paths[view_type].path, "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        self.offsets[view_type][row[1]] = {
                            "start_frame": int(row[2]),
                            "new_end_frame": int(row[3]) if row[3] != "-" else -1,
                        }

            self.dump_status = {}
            if isinstance(sequence_dataset_args, PathSource):
                self.sequence_dataset = pickle.load(
                    open(sequence_dataset_args.path, "rb")
                )
                # TODO (3): da implementare quando ci sarà il dump
                fine_support, fine_support_order = self.gen_supports(
                    self.tsn_dataset.video_list,
                    lambda item: item.path.split("/")[0],
                    lambda item: item.path.split("/")[1],
                    lambda item: item.start_frame,
                    lambda item: item.num_frames
                    + item.start_frame
                    - 1,  # 1 = modifier in gen_fine_labels.py
                )
                self.dump_status["sequence_dataset"] = {
                    "is_dump": True,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.AA_DUMP_LOAD.value,
                    "Loaded Action recognition features dump successfully",
                    DisplayLevel.HIGH,
                )
            else:
                self.sequence_dataset = aa_dataset.SequenceDataset(
                    **sequence_dataset_args
                )
                fine_support, fine_support_order = self.gen_supports(
                    self.sequence_dataset.video_list,
                    lambda item: item.path.split("/")[0],
                    lambda item: item.path.split("/")[1],
                    lambda item: item.start_frame,
                    lambda item: item.end_frame,
                )
                self.dump_status["sequence_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.AA_LOAD.value,
                    "Loaded Action recognition database successfully",
                    DisplayLevel.HIGH,
                )
            if isinstance(tsn_dataset_args, PathSource):
                self.tsn_dataset = pickle.load(open(tsn_dataset_args.path, "rb"))
                self.dump_status["tsn_dataset"] = {
                    "is_dump": True,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.AR_DUMP_LOAD.value,
                    "Loaded Action recognition features dump successfully",
                    DisplayLevel.HIGH,
                )
            else:
                self.tsn_dataset = ar_dataset.TSNDataSet(**tsn_dataset_args)
                self.dump_status["tsn_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.AR_LOAD.value,
                    "Loaded Action recognition database successfully",
                    DisplayLevel.HIGH,
                )

            if isinstance(augment_dataset_args, PathSource):
                self.augment_dataset = pickle.load(
                    open(augment_dataset_args.path, "rb")
                )
                # TODO (3): da implementare quando ci sarà il dump
                coarse_support, coarse_order_support = self.gen_supports(
                    self.augment_dataset.data,
                    lambda item: item["video_id"],
                    lambda item: item["view"],
                    lambda item: item["st_frame"],
                    lambda item: item["end_frame"],
                )
                self.dump_status["augment_dataset"] = {
                    "is_dump": True,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.TAS_DUMP_LOAD.value,
                    "Loaded Temporal action segmentation features dump successfully",
                    DisplayLevel.HIGH,
                )
            else:
                self.augment_dataset = tas_dataset.AugmentDataset(
                    **augment_dataset_args
                )
                coarse_support, coarse_order_support = self.gen_supports(
                    self.augment_dataset.data,
                    lambda item: item["video_id"],
                    lambda item: item["view"],
                    lambda item: item["st_frame"],
                    lambda item: item["end_frame"],
                    True,
                )
                self.dump_status["augment_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.TAS_LOAD.value,
                    "Loaded Temporal action segmentation database successfully",
                    DisplayLevel.HIGH,
                )

            if gaze_target is not None:
                if gaze_target.mode is SourceMode.DUMP:
                    if not os.path.isfile(gaze_target.path):
                        error_message = "Gaze target dump must be a file"
                        log_manager.log(
                            self.__class__.__name__,
                            LogCode.ERROR,
                            DatasetOpCode.GT_DUMP_LOAD.value,
                            error_message,
                            DisplayLevel.LOW,
                        )
                        raise ValueError(error_message)
                    self.gaze_target_dataset: dict | None = pickle.load(
                        open(gaze_target.path, "rb")
                    )
                    self.dump_status["gaze_target_dataset"] = {
                        "is_dump": True,
                        "is_support": False,
                    }
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        DatasetOpCode.GT_DUMP_LOAD.value,
                        "Loaded overall gaze target dump successfully",
                        DisplayLevel.HIGH,
                    )
                else:
                    error_message = "Gaze target must be a dump"
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        DatasetOpCode.GT_DUMP_LOAD.value,
                        error_message,
                        DisplayLevel.LOW,
                    )
                    raise ValueError(error_message)
            else:
                self.dump_status["gaze_target_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }
                self.gaze_target_dataset: dict | None = None

            if self.gaze_target_dataset is None:
                if (
                    gaze_source is not None
                    and os.path.isfile(gaze_source.path)
                    and gaze_source.mode is SourceMode.DUMP
                ):
                    self.gaze_source_dataset: dict[str, dict[str, str]] | PathSource = (
                        pickle.load(open(gaze_source.path, "rb"))
                    )
                    self.dump_status["gaze_source_dataset"] = {
                        "is_dump": True,
                        "is_support": False,
                    }
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        DatasetOpCode.GS_DUMP_LOAD.value,
                        "Loaded gaze source dump successfully",
                        DisplayLevel.HIGH,
                    )
                elif (
                    gaze_source is not None
                    and os.path.isdir(gaze_source.path)
                    and gaze_source.mode is SourceMode.VIDEO
                ):
                    self.gaze_source_dataset: dict[str, dict[str, str]] | PathSource = (
                        gaze_source
                    )
                    # self.gaze_source_dataset: dict[str, dict[str, str]] = {
                    #     sequence: {
                    #         view.replace(
                    #             ".mp4", ""
                    #         ): GazeObjectTracking.get_gaze_source_data(
                    #             PathSource(
                    #                 SourceMode.VIDEO,
                    #                 True,
                    #                 os.path.join(
                    #                     gaze_source.path, sequence, view
                    #                 ).replace("\\", "/"),
                    #                 False,
                    #                 gaze_source.params,
                    #             )
                    #         )
                    #         for view in os.listdir(
                    #             os.path.join(gaze_source.path, sequence)
                    #         )
                    #     }
                    #     for sequence in os.listdir(pose_source.path)
                    # }
                    self.dump_status["gaze_source_dataset"] = {
                        "is_dump": False,
                        "is_support": False,
                    }
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        DatasetOpCode.GS_DUMP_LOAD.value,
                        "Loaded gaze source successfully",
                        DisplayLevel.HIGH,
                    )
                else:
                    error_message = "Gaze source must be a dump file or a directory"
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        DatasetOpCode.GS_DUMP_LOAD.value,
                        error_message,
                        DisplayLevel.LOW,
                    )
                    raise ValueError(error_message)

                if not (
                    object_source is not None
                    and os.path.isfile(object_source.path)
                    and object_source.mode in [SourceMode.DUMP, SourceMode.SUPPORT_DUMP]
                ):
                    error_message = (
                        "Object source must be a dump file"
                        if object_source.mode is SourceMode.DUMP
                        else "Object source support dump must be a dump file"
                    )
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        DatasetOpCode.OS_DUMP_LOAD.value,
                        error_message,
                        DisplayLevel.LOW,
                    )
                    raise ValueError(error_message)
                self.object_source_dataset: dict | None = pickle.load(
                    open(object_source.path, "rb")
                )
                self.dump_status["object_source_dataset"] = {
                    "is_dump": True,
                    "is_support": object_source.mode is SourceMode.SUPPORT_DUMP,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    (
                        DatasetOpCode.OS_DUMP_LOAD.value
                        if object_source.mode is SourceMode.DUMP
                        else DatasetOpCode.OS_SUP_DUMP_LOAD.value
                    ),
                    f"Loaded object source {'support' if object_source.mode is SourceMode.SUPPORT_DUMP else ''} dump successfully",
                    DisplayLevel.HIGH,
                )
            else:
                self.gaze_source_dataset = None
                self.object_source_dataset = None
                self.dump_status["gaze_source_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }
                self.dump_status["object_source_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }

            if (
                self.gaze_target_dataset is None and self.object_source_dataset is None
            ):  # self.gaze_source_dataset not needed since it is None if gaze_target_dataset is None
                error_message = "Without gaze target dump at least object source support dump must be provided"
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    DatasetOpCode.OS_SUP_DUMP_LOAD.value,
                    error_message,
                    DisplayLevel.LOW,
                )
                raise ValueError(error_message)

            if (
                pose_source is not None
                and os.path.isfile(pose_source.path)
                and pose_source.mode is SourceMode.DUMP
            ):
                # TODO (3): da vedere list[float] in base a implementazione
                self.pose_source_dataset: (
                    dict[str, dict[str, list[float]]] | dict[str, str]
                ) = pickle.load(open(pose_source.path, "rb"))
                self.dump_status["pose_source_dataset"] = {
                    "is_dump": True,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.PS_DUMP_LOAD.value,
                    "Loaded pose source dump successfully",
                    DisplayLevel.HIGH,
                )
            elif (
                pose_source is not None
                and os.path.isdir(pose_source.path)
                and pose_source.mode is SourceMode.VIDEO
            ):
                self.pose_source_dataset: (
                    dict[str, dict[str, list[float]]] | PathSource
                ) = pose_source
                self.dump_status["pose_source_dataset"] = {
                    "is_dump": False,
                    "is_support": False,
                }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.PS_DUMP_LOAD.value,
                    "Loaded pose source successfully",
                    DisplayLevel.HIGH,
                )
            else:
                error_message = "Pose source must be a dump file or a directory"
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    DatasetOpCode.PS_DUMP_LOAD.value,
                    error_message,
                    DisplayLevel.LOW,
                )
                raise ValueError(error_message)

            self.skill_data: list[VideoRecord] = []
            self.datasets_refs: list[dict[str, list[int]]] = []
            with open(
                os.path.join(skill_splits_path.path, split.value + ".csv"),
                "r",
            ) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    item = VideoRecord(row[1:] + [row[0]])
                    self.skill_data.append(item)

                    sequence = item.sequence
                    view = item.view
                    start_frame = item.start_frame
                    end_frame = item.end_frame

                    tmp_ref = {
                        "aa_data": [],
                        "ar_data": [],
                        "tas_data": [],
                    }
                    # Takes data of Action Anticipation and Action Recognition datasets items contained in the range of start_frame and end_frame
                    support_i = (
                        fine_support[sequence][view][start_frame]["support_order_index"]
                        if start_frame in fine_support[sequence][view]
                        else 0
                    )
                    for i in range(support_i, len(fine_support_order[sequence][view])):
                        if (
                            fine_support_order[sequence][view][i] >= start_frame
                            and fine_support_order[sequence][view][i] < end_frame
                        ):
                            fine_item = fine_support[sequence][view][
                                fine_support_order[sequence][view][i]
                            ]
                            tmp_ref["aa_data"].append(fine_item["index"])
                            tmp_ref["ar_data"].append(fine_item["index"])

                            if fine_item["end_frame"] >= end_frame:
                                break
                        elif fine_support_order[sequence][view][i] > end_frame:
                            break

                    # Takes data of Temporal Action Segmentation dataset items contained in the range of start_frame and end_frame
                    support_i = (
                        coarse_support[sequence][view][start_frame][
                            "support_order_index"
                        ]
                        if start_frame in coarse_support[sequence][view]
                        else 0
                    )
                    for i in range(
                        support_i, len(coarse_order_support[sequence][view])
                    ):
                        coarse_item = coarse_support[sequence][view][
                            coarse_order_support[sequence][view][i]
                        ]
                        if (
                            coarse_order_support[sequence][view][i] >= start_frame
                            and coarse_order_support[sequence][view][i] < end_frame
                            or coarse_item["end_frame"] > start_frame
                            and coarse_item["end_frame"] <= end_frame
                        ):
                            tmp_ref["tas_data"].append(coarse_item["index"])

                        if coarse_item["end_frame"] >= end_frame:
                            break

                    self.datasets_refs.append(tmp_ref)
        """except Exception as e:
            self.gaze_source_dataset = None
            self.object_source_dataset = None
            self.gaze_target_dataset = None
            self.pose_source = None
            self.sequence_dataset = None
            self.tsn_dataset = None
            self.augment_dataset = None
            error_message = str(e) if debug_on else "Error loading dataset"
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                DatasetOpCode.DATASET_LOAD.value,
                error_message,
                DisplayLevel.LOW,
            )
            raise ValueError(error_message)"""

    def __len__(self):
        return len(self.skill_data)

    def __getitem__(self, idx: int):
        item = self.skill_data[idx]
        sequence = item.sequence
        view = item.view
        offset = self.offsets["ego"][f"{sequence}/{view}.mp4"][
            "start_frame"
        ]  # In this version "start_frame" is the same both for ego and fixed videos
        start_frame = self.map_frame(item.start_frame, offset)
        end_frame = self.map_frame(item.end_frame, offset)
        # if end_frame > self.offsets[view][sequence]["new_end_frame"]: # In this version "end_frame" after "new_end_frame" is already cleaned
        #     end_frame = self.offsets[view][sequence]["new_end_frame"]

        aa_data: list[dict[str, any]] = [
            self.sequence_dataset[i] for i in self.datasets_refs[idx]["aa_data"]
        ]
        ar_data: list[tuple[list[any], int]] = [
            self.tsn_dataset[i] for i in self.datasets_refs[idx]["ar_data"]
        ]
        tas_data: list[tuple[any]] = [
            self.augment_dataset.data[i] for i in self.datasets_refs[idx]["tas_data"]
        ]

        gaze_target_data = None
        gaze_source_data = None
        object_source_data = None
        if self.gaze_target_dataset is not None:
            gaze_target_data = [
                self.gaze_target_dataset[sequence][str(frame)]
                for frame in range(start_frame, end_frame)
            ]
        else:
            original_frames = []
            object_source_data = []
            for frame in range(start_frame, end_frame):
                frame_path = PathSource(
                    SourceMode.IMAGE,
                    False,
                    os.path.join(
                        self.gaze_source_dataset.path,
                        str(sequence),
                        view,
                        self.image_tmpl.format(view=view, frame=frame),
                    ).replace("\\", "/"),
                    False,
                    self.gaze_source_dataset.params,
                )
                if (
                    not self.dump_status["gaze_source_dataset"]["is_dump"]
                    and self.dump_status["object_source_dataset"]["is_support"]
                ):
                    original_frames.append(cv2.imread(frame_path.path))
                if (
                    not self.dump_status["object_source_dataset"]["is_support"]
                    or frame in self.object_source_dataset[sequence]
                ):
                    object_source_data.append(
                        self.object_source_dataset[sequence][frame]
                    )
                else:
                    object_source_data.append(None)
            gaze_source_data = (
                frame_path,
                original_frames,
            )

        if self.dump_status["pose_source_dataset"]["is_dump"]:
            pose_data: list[float] | list[cv2.typing.MatLike] = [
                self.pose_source_dataset[sequence][str(frame)]
                for frame in range(start_frame, end_frame)
            ]
        else:
            pose_data: list[float] | list[cv2.typing.MatLike] = [
                cv2.imread(
                    os.path.join(
                        self.pose_source_dataset.path,
                        str(sequence),
                        view,
                        self.image_tmpl.format(view=view, frame=frame),
                    ).replace("\\", "/")
                )
                for frame in range(start_frame, end_frame)
                for view in os.listdir(
                    os.path.join(self.pose_source_dataset.path, str(sequence))
                )
            ]

        return {
            "sequence": sequence,
            "view": view,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "aa_data": aa_data,
            "ar_data": ar_data,
            "tas_data": tas_data,
            "gaze_target_data": gaze_target_data,
            "gaze_source_data": gaze_source_data,
            "object_source_data": object_source_data,
            "pose_data": pose_data,
        }

    def map_frame(self, i, offset):
        return (i - offset) * self.target_fps // self.annotations_fps + 1

    def gen_supports(
        self,
        source: list,
        sequence_fn,
        view_fn,
        start_frame_fn,
        end_frame_fn,
        use_offset=False,
    ):
        support: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
        support_order: dict[str, dict[str, list[int]]] = {}
        for i, item in enumerate(source):
            sequence = sequence_fn(item)
            view = view_fn(item)
            offset = (
                self.offsets["ego"][f"{sequence}/{view}.mp4"]["start_frame"]
                if use_offset
                else 0
            )

            if sequence not in support:
                support[sequence] = {}
            if view not in support[sequence]:
                support[sequence][view] = {}

            if sequence not in support_order:
                support_order[sequence] = {}
            if view not in support_order[sequence]:
                support_order[sequence][view] = []

            start_frame = int(start_frame_fn(item)) + offset
            support[sequence][view][start_frame] = {
                "index": i,
                "support_order_index": len(support[sequence][view]),
                "end_frame": int(end_frame_fn(item)) + offset,
            }

            support_order[sequence][view].append(start_frame)

        for sequence in support_order:
            for view in support_order[sequence]:
                support_order[sequence][view].sort()

        return support, support_order


if __name__ == "__main__":
    # Test Combined Dataset
    import torchvision
    import pandas as pd

    from assembly101.temporal_action_segmentation.utils import dotdict
    from assembly101.action_recognition.dataset_config import return_dataset
    from assembly101.action_recognition.transforms import (
        Stack,
        ToTorchFormatTensor,
        GroupMultiScaleCrop,
        GroupRandomHorizontalFlip,
    )

    # Action anticipation dataset arguments
    aa_args = dotdict(
        {
            "mode": "train",
            "trainval": True,
            "path_to_data": "D:/data/TSM_features/",
            "path_to_anno": "D:/data/annotations/action_anticipation/",
            "path_to_models": "D:/data/models/action_anticipation",
            "path_to_offsets": "D:/data/annotations/ego_offsets.csv",
            "add_suffix": None,
            "task": "anticipation",
            "img_tmpl": "{:010d}.jpg",
            "resume": False,
            "best_model": "best",
            "modality": "ego",
            "views": "all",
            "num_workers": 0,
            "display_every": 1,
            "schedule_on": 1,
            "schedule_epoch": 10,
            "action_class": 1064,
            "verb_class": 17,
            "object_class": 90,
            "lr": 1e-4,
            "latent_dim": 512,
            "linear_dim": 512,
            "dropout_rate": 0.3,
            "scale_factor": -0.5,
            "scale": True,
            "batch_size": 32,
            "epochs": 15,
            "video_feat_dim": 2048,
            "past_attention": False,
            "spanning_sec": 6.0,
            "span_dim1": 5,
            "span_dim2": 3,
            "span_dim3": 2,
            "recent_dim": 2,
            "recent_sec1": 1.6,
            "recent_sec2": 1.2,
            "recent_sec3": 0.8,
            "recent_sec4": 0.4,
            "verb_object_scores": True,
            "add_verb_loss": True,
            "add_object_loss": True,
            "verb_loss_weight": 1.0,
            "object_loss_weight": 1.0,
            "topK": 1,
            "save_json": "D:/data/results/action_anticipation/",
            "model_name_ev": None,
            "alpha": 1.0,
            "debug_on": False,
        }
    )
    out_views = []
    if aa_args.views == "all":
        vs = list(view_dict["ego"].keys())
    else:
        vs = aa_args.views.strip(" ").split("+")
    out_v = []
    for x in vs:
        out_v.extend(view_dict["ego"][x])
    out_views.extend(out_v)
    aa_args.views = out_views.copy()
    sequence_dataset_args = dotdict(
        {
            "path_to_lmdb": "D:/data/TSM_features/",
            "path_to_csv": os.path.join(aa_args.path_to_anno, f"trainval.csv").replace(
                "\\", "/"
            ),
            "time_step": aa_args.alpha,
            "label_type": ["verb_id", "noun_id", "action_id"],
            "img_tmpl": aa_args.img_tmpl,
            "offsets_path": aa_args.path_to_offsets,
            "challenge": False,
            "args": aa_args,
        }
    )

    # Action recognition dataset arguments
    tsn_args = dotdict(
        {
            "dataset": "Assembly101",
            "num_segments": 3,
            "modality": "mono",
            "arch": "resnet50",
            "dense_sample": False,
        }
    )
    (
        num_class,
        tsn_args.train_list,
        tsn_args.val_list,
        tsn_args.root_path,
        tsn_prefix,
    ) = return_dataset(tsn_args.dataset, tsn_args.modality)
    if tsn_args.modality == "mono" or tsn_args.modality == "combined":
        tsn_args.modality = "RGB"
    tsn_dataset_args = dotdict(
        {
            "root_path": tsn_args.root_path,
            "list_file": tsn_args.train_list,
            "num_segments": tsn_args.num_segments,
            "new_length": 1,
            "modality": tsn_args.modality,
            "image_tmpl": tsn_prefix,
            "transform": torchvision.transforms.Compose(
                [
                    torchvision.transforms.Compose(
                        [
                            GroupMultiScaleCrop(224, [1, 0.875, 0.75, 0.66]),
                            GroupRandomHorizontalFlip(is_flow=False),
                        ]
                    ),
                    Stack(roll=(tsn_args.arch in ["BNInception", "InceptionV3"])),
                    ToTorchFormatTensor(
                        div=(tsn_args.arch not in ["BNInception", "InceptionV3"])
                    ),
                ]
            ),
            "dense_sample": tsn_args.dense_sample,
        }
    )

    # Temporal action segmentation dataset arguments
    views = []
    for view in view_dict["ego"]:
        views.extend(view_dict["ego"][view])
    tas_config = dotdict(
        epochs=200,
        dataset="assembly",
        feature_size=2048,
        gamma=0.5,
        step_size=200,
        split="train_val",
        gt_path="D:/data/annotations/coarse-annotations/coarse_labels/",
        features_path="D:/data/TSM_features/",
        VIEWS=views,
    )
    if tas_config.dataset == "assembly":
        tas_config.lmdb_fps = 15
        tas_config.annotations_fps = 30
        tas_config.chunk_size = 20 * tas_config.lmdb_fps // tas_config.annotations_fps
        tas_config.max_frames_per_video = (
            1200 * tas_config.lmdb_fps // tas_config.annotations_fps
        )
        tas_config.learning_rate = 1e-4
        tas_config.weight_decay = 1e-4
        tas_config.batch_size = 32
        tas_config.num_class = 169
        tas_config.back_gd = []
        tas_config.ensem_weights = [1, 1, 1, 1, 1, 1]
    else:
        print("not defined yet")
        exit(1)
    coarse_actions = pd.read_csv(
        "D:/data/annotations/coarse-annotations/actions.csv",
        header=0,
        names=["action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls"],
    )
    coarse_actions_dict = dict()
    for _, act in coarse_actions.iterrows():
        coarse_actions_dict[act["action_cls"]] = int(act["action_id"])
    augment_dataset_args = {
        "args": tas_config,
        "fold": "train",
        "fold_file_name": "D:/data/annotations/coarse-annotations/coarse_splits/",
        "actions_dict": coarse_actions_dict,
        "zoom_crop": (0.5, 2),
        "offsets_path": "D:/data/annotations/ego_offsets.csv",
    }

    gaze_source_path = PathSource(
        SourceMode.VIDEO,
        False,
        "D:/data/ego_recordings",
        False,
        {
            "proximity_threshold": 40,
            "approaching_threshold": 60,
            "coordinates_memory_size": 15,
            "correct_directions": 10,
        },
    )

    object_source_path = PathSource(
        SourceMode.SUPPORT_DUMP,
        False,
        "D:/data/dumps/gaze_analysis/instructions_annotations.pkl",
        False,
    )

    pose_source_path = PathSource(
        SourceMode.VIDEO,
        False,
        "D:/data/fixed_recordings",
        False,
    )

    offsets_paths = {
        view_type: PathSource(
            SourceMode.DUMP,
            False,
            f"D:/data/annotations/{view_type}_offsets.csv",
            False,
        )
        for view_type in view_dict.keys()
    }

    skill_splits_path = PathSource(
        SourceMode.SPLIT,
        False,
        "D:/data/annotations/skill_evaluation/skill_splits/",
        False,
    )

    dataset = CombinedDataset(
        ComposedSplit.TRAINVAL,
        30,
        15,
        offsets_paths,
        skill_splits_path,
        "{view}_{frame:010d}.jpg",
        sequence_dataset_args,
        tsn_dataset_args,
        augment_dataset_args,
        pose_source_path,
        object_source_path,
        gaze_source_path,
    )

    a = dataset[0]
