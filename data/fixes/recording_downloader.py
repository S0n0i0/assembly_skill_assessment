import win32com.client
import os
import shutil
import ffmpeg
from tqdm import tqdm

shell = win32com.client.Dispatch("WScript.Shell")

lnk_path = "G:/Il mio Drive/recordings.lnk"
no_proper_videos_file_path = "no_proper_viedos.txt"
decrease = True
mode = 1
if mode == 0:
    target_path = "D:/data/ego_recordings"
    offsets_file_path = "D:/data/annotations/ego_offsets.csv"
    skip_directories = []
    is_proper = lambda f: f.find("HMC_") != -1
elif mode == 1:
    target_path = "D:/data/fixed_recordings"
    offsets_file_path = "D:/data/annotations/fixed_offsets.csv"
    skip_directories = []
    is_proper = lambda f: f == "C10118_rgb.mp4"

shortcut = shell.CreateShortCut(lnk_path)
drive_path = shortcut.Targetpath

dirs = os.listdir(drive_path)
os.chdir(drive_path)

offsets = {}
if offsets_file_path != "":
    with open(offsets_file_path, "r") as f:
        offsets = {}
        for line in f.readlines()[1:]:
            line = line.strip().split(",")
            sequence = line[1].split("/")[0]
            view = line[1].split("/")[1]
            if sequence not in offsets:
                offsets[sequence] = {}
            offsets[sequence][view] = int(line[2])

with open(target_path + "/" + no_proper_videos_file_path, "a") as no_proper_videos:
    count = 0
    for i, sequence in enumerate(dirs):
        if i in skip_directories:
            print("Skipped directory n. " + str(i) + " (" + sequence + ")")
        elif sequence not in offsets:
            print("Sequence n. " + str(i) + " (" + sequence + ") not in offsets")
        else:
            target_dir = target_path + "/" + sequence
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
                exist_target = False
            else:
                exist_target = True

            files = [f for f in os.listdir(sequence) if is_proper(f)]
            if exist_target:
                present_files = os.listdir(target_dir)
                print(
                    "Files already present in directory n. "
                    + str(i)
                    + " ("
                    + sequence
                    + "):",
                    present_files,
                )
                files = [f for f in files if f not in present_files]

            if len(files) == 0 and (not exist_target or len(present_files) == 0):
                print(
                    "The directory n. "
                    + str(i)
                    + " ("
                    + sequence
                    + ") has no proper videos"
                )
                no_proper_videos.write(sequence + "\n")
            elif len(files) > 0:
                print(
                    "Copying directory n. "
                    + str(i)
                    + "/"
                    + str(len(dirs) - 1)
                    + " ["
                    + str(count)
                    + "/"
                    + str(len(offsets) - 1)
                    + "] ("
                    + sequence
                    + ")"
                )
                count += 1
                for f in tqdm(files):
                    target_file_path = target_dir + "/" + f
                    source_file_path = sequence + "/" + f
                    if decrease:
                        common = ffmpeg.input(source_file_path)
                        if mode == 1:
                            common = common.filter("scale", width=640, height=360)
                        common.filter("fps", fps=15, round="up").output(
                            target_file_path, vcodec="h264_nvenc", loglevel="quiet"
                        ).run()
                    else:
                        shutil.copyfile(source_file_path, target_file_path)
            else:
                count += 1

print("Finish")
