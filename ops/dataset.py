from torch.utils.data import Dataset
from enum import Enum
import os
import pickle

import assembly101.action_anticipation.dataset as aa_dataset
import assembly101.action_recognition.dataset as ar_dataset
import assembly101.temporal_action_segmentation.dataset as tas_dataset

from utils.classes import PathSource, SourceMode
from utils.enums import (
    LogCode,
    DisplayLevel,
)
from utils.constants import debug_on, log_manager


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
        sequence_dataset_args: PathSource | dict,
        tsn_dataset_args: PathSource | dict,
        augment_dataset_args: PathSource | dict,
        object_source: PathSource | None = None,
        gaze_source: PathSource | None = None,
        gaze_target: PathSource | None = None,
        pose_source: PathSource | None = None,
    ):
        try:
            if isinstance(sequence_dataset_args, PathSource):
                self.sequence_dataset = pickle.load(
                    open(sequence_dataset_args.path, "rb")
                )
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
                self.fine_support, self.fine_support_order = self.gen_supports(
                    self.sequence_dataset.video_list,
                    lambda item: item.path.split("/")[0],
                    lambda item: item.path.split("/")[1],
                    lambda item: item.start_frame,
                    lambda item: item.end_frame,
                )
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.AA_LOAD.value,
                    "Loaded Action recognition database successfully",
                    DisplayLevel.HIGH,
                )
            if isinstance(tsn_dataset_args, PathSource):
                self.tsn_dataset = pickle.load(open(tsn_dataset_args.path, "rb"))
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.AR_DUMP_LOAD.value,
                    "Loaded Action recognition features dump successfully",
                    DisplayLevel.HIGH,
                )
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
                self.coarse_support, self.coarse_order_support = self.gen_supports(
                    self.augment_dataset.data,
                    lambda item: item["video_id"],
                    lambda item: item["view"],
                    lambda item: item["st_frame"],
                    lambda item: item["end_frame"],
                )
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
                    self.gaze_target: dict | None = pickle.load(
                        open(gaze_target.path, "rb")
                    )
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
                self.gaze_target: dict | None = None

            if self.gaze_target is None:
                if gaze_source is not None:
                    if not (
                        os.path.isfile(gaze_source.path)
                        and gaze_source.mode is SourceMode.DUMP
                    ):
                        error_message = "Gaze source must be a dump file"
                        log_manager.log(
                            self.__class__.__name__,
                            LogCode.ERROR,
                            DatasetOpCode.GS_DUMP_LOAD.value,
                            error_message,
                            DisplayLevel.LOW,
                        )
                        raise ValueError(error_message)
                    self.gaze_source: dict | None = pickle.load(
                        open(gaze_source.path, "rb")
                    )
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        DatasetOpCode.GS_DUMP_LOAD.value,
                        "Loaded gaze source dump successfully",
                        DisplayLevel.HIGH,
                    )

                if object_source is not None:
                    if not (
                        os.path.isfile(object_source.path)
                        and object_source.mode
                        in [SourceMode.DUMP, SourceMode.SUPPORT_DUMP]
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
                    self.object_source: dict | None = pickle.load(
                        open(object_source.path, "rb")
                    )
                    log_manager.log(
                        self.__class__.__name,
                        LogCode.SUCCESS,
                        (
                            DatasetOpCode.OS_DUMP_LOAD.value
                            if object_source.mode is SourceMode.DUMP
                            else DatasetOpCode.OS_SUP_DUMP_LOAD.value
                        ),
                        f"Loaded object source {'support' if object_source.mode is SourceMode.SUPPORT_DUMP else ''} dump successfully",
                        DisplayLevel.HIGH,
                    )

            """if self.gaze_target is None and object_source is None:
                error_message = "Without gaze target dump at least object source support dump must be provided"
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    DatasetOpCode.OS_SUP_DUMP_LOAD.value,
                    error_message,
                    DisplayLevel.LOW,
                )
                raise ValueError(error_message)"""

            if pose_source is not None:
                if not (
                    os.path.isfile(pose_source.path)
                    and pose_source.mode is SourceMode.DUMP
                ):
                    error_message = "Pose source must be a dump file"
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        DatasetOpCode.PS_DUMP_LOAD.value,
                        error_message,
                        DisplayLevel.LOW,
                    )
                    raise ValueError(error_message)
                self.pose_source: dict = pickle.load(open(pose_source.path, "rb"))
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    DatasetOpCode.PS_DUMP_LOAD.value,
                    "Loaded pose source dump successfully",
                    DisplayLevel.HIGH,
                )

        except Exception as e:
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
                DatasetOpCode.DATASET_LOAD.value,
                str(e) if debug_on else "Error loading dataset",
                DisplayLevel.LOW,
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
            "trainval": False,
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
    view_dict = {
        "ego": {
            "view1": ["HMC_21176875_mono10bit", "HMC_84346135_mono10bit"],
            "view2": ["HMC_21176623_mono10bit", "HMC_84347414_mono10bit"],
            "view3": ["HMC_21110305_mono10bit", "HMC_84355350_mono10bit"],
            "view4": ["HMC_21179183_mono10bit", "HMC_84358933_mono10bit"],
        },
    }
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
            "path_to_lmdb": "D:/data/assembly/TSM_features/",
            "path_to_csv": os.path.join(aa_args.path_to_anno, f"trainval.csv"),
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

    dataset = CombinedDataset(
        sequence_dataset_args,
        tsn_dataset_args,
        augment_dataset_args,
    )
