# Code for TSM adapted from the original TSM repo:
# https://github.com/mit-han-lab/temporal-shift-module

import os
import os.path
import numpy as np
from numpy.random import randint
from torch.utils import data
from PIL import Image
import cv2
import time
from collections import OrderedDict
from multiprocessing import Lock


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(
        self,
        root_path,
        list_file,
        num_segments=3,
        new_length=1,
        modality="RGB",
        image_tmpl="frame_{:010d}.jpg",
        transform=None,
        force_grayscale=False,
        random_shift=True,
        dense_sample=False,
        test_mode=False,
    ):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.use_video = True
        self.cache = OrderedDict()
        self.max_size = 10
        self.timeout = 300
        # create a lock
        self.read_lock = Lock()
        self.write_lock = Lock()

        if self.modality == "RGBDiff":
            # Diff needs one more image to calculate diff
            self.new_length += 1

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == "RGB" or self.modality == "RGBDiff":
            # print(os.path.join(directory, self.image_tmpl.format(idx)))

            return [
                Image.open(
                    os.path.join(
                        directory,
                        directory.split("/")[-1] + "_" + self.image_tmpl.format(idx),
                    )
                ).convert("RGB")
            ]
            # return [Image.new('RGB', (456, 256), (73, 109, 137))]
        elif self.modality == "Flow":
            x_img = Image.open(
                os.path.join(
                    directory,
                    directory.split("/")[-1] + "_" + self.image_tmpl.format("x", idx),
                )
            ).convert("L")
            y_img = Image.open(
                os.path.join(
                    directory,
                    directory.split("/")[-1] + "_" + self.image_tmpl.format("y", idx),
                )
            ).convert("L")

            return [x_img, y_img]

    def _load_video_image(self, video, idx, framerate_ratio=4):
        if self.modality == "RGB" or self.modality == "RGBDiff":
            actual_idx = idx // framerate_ratio
            video.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
            _, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            return [frame]
        else:
            raise ValueError("Not implemented yet")

    def _parse_list(self):
        tmp = [x.strip().split(" ") for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
        self.video_list = [VideoRecord(x) for x in tmp]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (
            record.num_frames - self.new_length + 1
        ) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration
            ) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(
                randint(record.num_frames - self.new_length + 1, size=self.num_segments)
            )
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            )
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
        )
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = (
                self._sample_indices(record)
                if self.random_shift
                else self._get_val_indices(record)
            )
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def _remove_oldest_video(self):
        oldest_path = next(iter(self.cache))
        self._close_video(oldest_path)

    def _close_video(self, path):
        self.cache[path]["video"].release()
        del self.cache[path]

    def close_all(self):
        for path in list(self.cache.keys()):
            self._close_video(path)

    def _remove_expired_videos(self, current_time):
        expired_paths = [
            path
            for path, data in self.cache.items()
            if current_time - data["last_access"] > self.timeout
        ]
        for path in expired_paths:
            self._close_video(path)

    def _remove_expired_videos(self, current_time):
        expired_paths = [
            path
            for path, data in self.cache.items()
            if current_time - data["last_access"] > self.timeout
        ]
        for path in expired_paths:
            self._close_video(path)

    def _remove_oldest_video(self):
        oldest_path = next(iter(self.cache))
        self._close_video(oldest_path)

    def _close_video(self, path):
        self.cache[path]["video"].release()
        del self.cache[path]

    def close_all(self):
        with self.write_lock:
            for path in list(self.cache.keys()):
                self._close_video(path)

    def get_processed_frames(self, video, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.new_length):
                seg_imgs = (
                    self._load_video_image(video, p)
                    if self.use_video
                    else self._load_image(record.path, p)
                )
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        return self.transform(images)

    def get(self, record, indices):
        if self.use_video:
            current_time = time.time()
            with self.read_lock:
                if record.path in self.cache:
                    self.cache[record.path]["last_access"] = current_time
                    cap = self.cache[record.path]["video"]
                else:
                    with self.write_lock:
                        # Remove expired videos
                        self._remove_expired_videos(current_time)
                        if len(self.cache) >= self.max_size:
                            self._remove_oldest_video()

                        cap = cv2.VideoCapture(record.path + ".mp4")
                        self.cache[record.path] = {
                            "video": cap,
                            "last_access": current_time,
                        }
                process_data = self.get_processed_frames(cap, record, indices)
        else:
            process_data = self.get_processed_frames(cap, record, indices)

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
