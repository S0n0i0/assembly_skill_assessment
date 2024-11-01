import torch
import lmdb
import pickle as pkl
import os
import numpy as np


################## Dataloader
def collate_fn_override(data):
    """
    data:
    """
    data = list(filter(lambda x: x is not None, data))
    (
        data_arr,
        count,
        labels,
        clip_length,
        start,
        video_id,
        labels_present_arr,
        aug_chunk_size,
    ) = zip(*data)

    return (
        torch.stack(data_arr),
        torch.tensor(count),
        torch.stack(labels),
        torch.tensor(clip_length),
        torch.tensor(start),
        video_id,
        torch.stack(labels_present_arr),
        torch.tensor(aug_chunk_size, dtype=torch.int),
    )


def get_offsets(offsets_path):
    offsets = {}
    with open(offsets_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        _, file, offset = line.strip().split(",")
        sequence, view = file.split("/")
        if sequence not in offsets:
            offsets[sequence] = {}
        offsets[sequence][view[:-4]] = int(offset)
    return offsets


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        fold,
        fold_file_name,
        actions_dict,
        zoom_crop=(0.5, 2),
        smallest_cut=1.0,
        offsets_path=None,
    ):
        self.fold = fold
        self.max_frames_per_video = args.max_frames_per_video
        self.feature_size = args.feature_size
        self.base_dir_name = args.features_path
        self.frames_format = "/{}/{}_{:010d}.jpg"
        self.ground_truth_files_dir = args.gt_path
        self.chunk_size = args.chunk_size
        self.num_class = args.num_class
        self.zoom_crop = zoom_crop
        self.validation = True if fold == "validation" else False
        self.split = args.split
        self.VIEWS = args.VIEWS
        self.actions_dict = actions_dict
        self.lmdb_fps = args.lmdb_fps
        self.annotations_fps = args.annotations_fps
        self.offsets = get_offsets(offsets_path) if offsets_path is not None else None

        with open(
            "D:/data/models/temporal_action_segmentation/statistic_input.pkl",
            "rb",
        ) as f:
            self.statistic = pkl.load(f)
        self.data = self.make_data_set(fold_file_name)

    def open_lmdb(self):
        self.env = {
            view: lmdb.open(f"{self.base_dir_name}/{view}", readonly=True, lock=False)
            for view in self.VIEWS
        }

    def map_frame(self, annotation_frame, offset):
        return (annotation_frame - offset) * self.lmdb_fps // self.annotations_fps + 1

    def read_files(self, list_files, fold_file_name):
        data = []
        for file in list_files:
            lines = open(fold_file_name + file).readlines()
            for l in lines:
                data.append(l.split("\t")[0])
        return data

    def make_data_set(self, fold_file_name):
        label_name_to_label_id_dict = self.actions_dict
        if self.fold == "train":
            if self.split == "train_val":
                files = [
                    "train_coarse_assembly.txt",
                    "validation_coarse_assembly.txt",
                ]
            elif self.split == "train":
                files = ["train_coarse_assembly.txt"]
        elif self.fold == "validation":
            files = ["validation_coarse_assembly.txt"]
        else:
            print("unknown split, quit")
            exit(1)

        data = self.read_files(files, fold_file_name)
        data_arr = []
        for i, video_id in enumerate(data):
            video_id = video_id.split(".txt")[0]
            sequence = video_id.replace("disassembly_", "").replace("assembly_", "")
            filename = os.path.join(self.ground_truth_files_dir, video_id + ".txt")

            recog_content, indexs = [], []
            offset = (
                self.offsets[sequence][list(self.offsets[sequence].keys())[0]]
                if self.offsets is not None
                else 0
            )
            with open(filename, "r") as f:
                lines = f.readlines()
                for l in lines:
                    tmp = l.split("\t")
                    start_l, end_l, label_l = (
                        self.map_frame(int(tmp[0]), offset),
                        self.map_frame(int(tmp[1]), offset),
                        tmp[2],
                    )
                    indexs.extend([start_l, end_l])
                    recog_content.extend([label_l] * (end_l - start_l))

            recog_content = [label_name_to_label_id_dict[e] for e in recog_content]
            span = [min(indexs), max(indexs)]  # [start end)

            total_frames = len(recog_content)
            assert total_frames == (span[1] - span[0])

            for view in self.VIEWS:
                type_action = video_id.split("_")[0]
                key_id = video_id.split(type_action)[1][1:]

                if key_id not in self.statistic or view not in self.statistic[key_id]:
                    continue
                # span[0] = max(span[0], self.statistic[key_id][view][0])
                assert self.statistic[key_id][view][0] <= span[0]
                span[1] = min(span[1], self.statistic[key_id][view][1])
                if span[1] <= span[0]:
                    # the video only involves preparation, no action before it's end.
                    continue

                start_frame_arr = []
                end_frame_arr = []
                for st in range(
                    span[0], span[1], self.max_frames_per_video * self.chunk_size
                ):
                    start_frame_arr.append(st)
                    max_end = st + (self.max_frames_per_video * self.chunk_size)
                    end_frame = max_end if max_end < span[1] else span[1]
                    end_frame_arr.append(end_frame)

                # print(span[1] - span[0])
                # if len(start_frame_arr) >= 2:
                #     print(video_id, view)

                for st_frame, end_frame in zip(start_frame_arr, end_frame_arr):
                    ele_dict = {
                        "type": type_action,
                        "view": view,
                        "st_frame": st_frame,
                        "end_frame": end_frame,
                        "video_id": key_id,
                        "tot_frames": (end_frame - st_frame),
                    }

                    ele_dict["labels"] = np.array(
                        recog_content[st_frame - span[0] : end_frame - span[0]],
                        dtype=int,
                    )
                    data_arr.append(ele_dict)

        print(
            "Number of videos logged in {} fold is {}".format(self.fold, len(data_arr))
        )
        return data_arr

    def getitem(self, index):  # Try to use this for debugging purpose
        if not hasattr(self, "env"):
            self.open_lmdb()

        ele_dict = self.data[index]
        st_frame = ele_dict["st_frame"]
        end_frame = ele_dict["end_frame"]
        sequence = ele_dict["video_id"]
        view = ele_dict["view"]
        vid_type = ele_dict["type"]

        elements = []
        with self.env[view].begin() as e:
            for i in range(st_frame, end_frame):
                key = sequence + self.frames_format.format(view, view, i)
                data = e.get(key.strip().encode("utf-8"))
                if data is None:
                    print("no available data.")
                    exit(2)
                data = np.frombuffer(data, "float32")
                assert data.shape[0] == 2048
                elements.append(data)

        elements = np.array(elements).T

        count = 0
        end_frame = min(
            end_frame, st_frame + (self.max_frames_per_video * self.chunk_size)
        )
        len_video = end_frame - st_frame

        if np.random.randint(low=0, high=2) == 0 and (not self.validation):
            min_possible_chunk_size = np.ceil(len_video / self.max_frames_per_video)
            max_chunk_size = int(1.0 * self.chunk_size / self.zoom_crop[0])
            min_chunk_size = max(
                int(1.0 * self.chunk_size / self.zoom_crop[1]), min_possible_chunk_size
            )
            aug_chunk_size = int(
                np.exp(
                    np.random.uniform(
                        low=np.log(min_chunk_size), high=np.log(max_chunk_size)
                    )
                )
            )
            num_aug_frames = np.ceil(int(len_video / aug_chunk_size))
            if num_aug_frames > self.max_frames_per_video:
                num_aug_frames = self.max_frames_per_video
                aug_chunk_size = int(np.ceil(len_video / num_aug_frames))

            aug_start_frame = st_frame
            aug_end_frame = end_frame
        else:
            aug_start_frame, aug_end_frame, aug_chunk_size = (
                st_frame,
                end_frame,
                self.chunk_size,
            )

        data_arr = torch.zeros((self.max_frames_per_video, self.feature_size))
        label_arr = torch.ones(self.max_frames_per_video, dtype=torch.long) * -100
        labels_present_arr = torch.zeros(self.num_class, dtype=torch.float32)
        for i in range(aug_start_frame, aug_end_frame, aug_chunk_size):
            end = min(aug_end_frame, i + aug_chunk_size)
            key = elements[:, i - aug_start_frame : end - aug_start_frame]
            values, counts = np.unique(
                ele_dict["labels"][i - aug_start_frame : end - aug_start_frame],
                return_counts=True,
            )
            label_arr[count] = values[np.argmax(counts)]
            labels_present_arr[label_arr[count]] = 1
            data_arr[count, :] = torch.tensor(np.max(key, axis=-1), dtype=torch.float32)
            count += 1

        return (
            data_arr,
            count,
            label_arr,
            ele_dict["tot_frames"],
            st_frame,
            vid_type + "_" + ele_dict["video_id"] + "%{}".format(view),
            labels_present_arr,
            aug_chunk_size,
        )

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


def collate_fn_override_test(data):
    """
    data:
    """
    data = list(filter(lambda x: x is not None, data))
    (
        data_arr,
        count,
        labels,
        video_len,
        start,
        video_id,
        labels_present_arr,
        chunk_size,
        chunk_id,
    ) = zip(*data)
    return (
        torch.stack(data_arr),
        torch.tensor(count),
        torch.stack(labels),
        torch.tensor(video_len),
        torch.tensor(start),
        video_id,
        torch.stack(labels_present_arr),
        torch.tensor(chunk_size),
        torch.tensor(chunk_id),
    )


class AugmentDataset_test(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        fold,
        fold_file_name,
        actions_dict,
        chunk_size,
        annotations_fps=30,
        lmdb_fps=15,
        offsets_path=None,
    ):
        self.fold = fold
        self.max_frames_per_video = args.max_frames_per_video
        self.feature_size = args.feature_size
        self.base_dir_name = args.features_path
        self.frames_format = "/{}/{}_{:010d}.jpg"
        self.ground_truth_files_dir = args.gt_path
        self.num_class = args.num_class
        self.VIEWS = args.VIEWS
        self.actions_dict = actions_dict
        self.lmdb_fps = lmdb_fps
        self.annotations_fps = annotations_fps
        self.offsets = get_offsets(offsets_path) if offsets_path is not None else None
        with open(
            "D:/data/models/temporal_action_segmentation/statistic_input.pkl",
            "rb",
        ) as f:
            self.statistic = pkl.load(f)
        self.chunk_size_arr = chunk_size
        self.data = self.make_data_set(fold_file_name)

    def open_lmdb(self):
        self.env = {
            view: lmdb.open(f"{self.base_dir_name}/{view}", readonly=True, lock=False)
            for view in self.VIEWS
        }
        print("Ciaone")

    def read_files(self, list_files, fold_file_name):
        data = []
        for file in list_files:
            lines = open(fold_file_name + file).readlines()
            for l in lines:
                data.append(l.split("\t")[0])
        return data

    def make_data_set(self, fold_file_name):
        label_name_to_label_id_dict = self.actions_dict
        if self.fold == "validation":
            files = ["validation_coarse_assembly.txt"]
        elif self.fold == "test":
            files = ["test_coarse_assembly.txt"]
        else:
            print("Unknown data folder")
            exit(3)
        data = self.read_files(files, fold_file_name)

        data_arr = []
        for i, video_id in enumerate(data):
            video_id = video_id.split(".txt")[0]
            if "disassembly" in video_id:
                video_id = video_id.replace("disassembly", "disassebly")
            filename = os.path.join(self.ground_truth_files_dir, video_id + ".txt")

            recog_content, indexs = [], []
            with open(filename, "r") as f:
                lines = f.readlines()
                for l in lines:
                    tmp = l.split("\t")
                    start_l, end_l, label_l = int(tmp[0]), int(tmp[1]), tmp[2]
                    indexs.extend([start_l, end_l])
                    recog_content.extend([label_l] * (end_l - start_l))

            recog_content = [label_name_to_label_id_dict[e] for e in recog_content]
            span = [min(indexs), max(indexs)]  # [start end)

            len_video = len(recog_content)
            assert len_video == (span[1] - span[0])

            chunk_size_arr = self.chunk_size_arr
            for view in self.VIEWS:
                type_action = video_id.split("_")[0]
                key_id = video_id.split(type_action)[1][1:]

                if view not in self.statistic[key_id]:
                    continue
                # span[0] = max(span[0], self.statistic[key_id][view][0])
                assert self.statistic[key_id][view][0] <= span[0]
                span[1] = min(span[1], self.statistic[key_id][view][1])
                if span[1] <= span[0]:
                    continue
                for j, chunk_size in enumerate(chunk_size_arr):
                    start_frame_arr = []
                    end_frame_arr = []
                    for st in range(
                        span[0], span[1], self.max_frames_per_video * chunk_size
                    ):
                        start_frame_arr.append(st)
                        max_end = st + (self.max_frames_per_video * chunk_size)
                        end_frame = max_end if max_end < span[1] else span[1]
                        end_frame_arr.append(end_frame)

                    # print(span[1] - span[0])
                    # if len(start_frame_arr) >= 2:
                    #     print(video_id, view)

                    for st_frame, end_frame in zip(start_frame_arr, end_frame_arr):
                        ele_dict = {
                            "type": type_action,
                            "view": view,
                            "st_frame": st_frame,
                            "end_frame": end_frame,
                            "chunk_id": j,
                            "chunk_size": chunk_size,
                            "video_id": key_id,
                            "tot_frames": (end_frame - st_frame) // chunk_size,
                        }

                        ele_dict["labels"] = np.array(
                            recog_content[st_frame - span[0] : end_frame - span[0]],
                            dtype=int,
                        )
                        data_arr.append(ele_dict)

        print(
            "Number of datapoints logged in {} fold is {}".format(
                self.fold, len(data_arr)
            )
        )
        return data_arr

    def getitem(self, index):  # Try to use this for debugging purpose
        if not hasattr(self, "env"):
            self.open_lmdb()

        ele_dict = self.data[index]
        st_frame = ele_dict["st_frame"]
        end_frame = ele_dict["end_frame"]
        aug_chunk_size = ele_dict["chunk_size"]
        sequence = ele_dict["video_id"]
        view = ele_dict["view"]
        vid_type = ele_dict["type"]
        offset = self.offsets[sequence][view] if self.offsets is not None else 0

        elements = []
        with self.env[view].begin() as e:
            for i in range(st_frame, end_frame):
                key = ele_dict["video_id"] + self.frames_format.format(
                    view, view, (i - offset) * self.lmdb_fps / self.annotations_fps
                )
                data = e.get(key.strip().encode("utf-8"))
                if data is None:
                    print("no available data.")
                    exit(2)
                data = np.frombuffer(data, "float32")
                assert data.shape[0] == 2048
                elements.append(data)

        elements = np.array(elements).T

        count = 0
        labels_present_arr = torch.zeros(self.num_class, dtype=torch.float32)
        data_arr = torch.zeros((self.max_frames_per_video, self.feature_size))
        label_arr = torch.ones(self.max_frames_per_video, dtype=torch.long) * -100
        for i in range(st_frame, end_frame, aug_chunk_size):
            end = min(end_frame, i + aug_chunk_size)
            key = elements[:, i - st_frame : end - st_frame]
            values, counts = np.unique(
                ele_dict["labels"][i - st_frame : end - st_frame], return_counts=True
            )
            label_arr[count] = values[np.argmax(counts)]
            labels_present_arr[label_arr[count]] = 1
            data_arr[count, :] = torch.tensor(np.max(key, axis=-1), dtype=torch.float32)
            count += 1

        return (
            data_arr,
            count,
            label_arr,
            elements.shape[1],
            st_frame,
            vid_type + "_" + ele_dict["video_id"] + "%{}".format(view),
            labels_present_arr,
            aug_chunk_size,
            ele_dict["chunk_id"],
        )

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)
