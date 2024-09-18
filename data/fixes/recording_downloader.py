import win32com.client
import os
import shutil

shell = win32com.client.Dispatch("WScript.Shell")

lnk_path = ""
no_proper_videos_file_path = "no_proper_viedos.txt"
mode = 0
if mode == 0:
    target_path = ""
    skip_directories = []
    is_proper = lambda f: f.find("HMC_") != -1
elif mode == 1:
    target_path = ""
    skip_directories = []
    is_proper = lambda f: f == "C10118_rgb.mp4"

shortcut = shell.CreateShortCut(lnk_path)
drive_path = shortcut.Targetpath

dirs = os.listdir(drive_path)
os.chdir(drive_path)

with open(target_path + "/" + no_proper_videos_file_path, "a") as no_proper_videos:
    for i, source_dir in enumerate(dirs):
        if i not in skip_directories:
            target_dir = target_path + "/" + source_dir
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
                exist_target = False
            else:
                exist_target = True

            files = [f for f in os.listdir(source_dir) if is_proper(f)]
            if exist_target:
                present_files = os.listdir(target_dir)
                print(
                    "Files already present in directory n. "
                    + str(i)
                    + " ("
                    + source_dir
                    + "):",
                    present_files,
                )
                files = [f for f in files if f not in present_files]

            if len(files) == 0 and (not exist_target or len(present_files) == 0):
                print(
                    "The directory n. "
                    + str(i)
                    + " ("
                    + source_dir
                    + ") has no proper videos"
                )
                no_proper_videos.write(source_dir + "\n")
            elif len(files) > 0:
                print("Copying directory n. " + str(i) + " (" + source_dir + ")")
                for f in files:
                    target_file_path = target_dir + "/" + f
                    source_file_path = source_dir + "/" + f
                    shutil.copyfile(source_file_path, target_file_path)
        else:
            print("Skipped directory n. " + str(i) + " (" + source_dir + ")")

print("Finish")
