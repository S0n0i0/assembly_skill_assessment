# NOT WORKING

import os
import paramiko.client
import threading
from collections import defaultdict
from tqdm import tqdm
import signal
import sys

threads = []


def signal_handler(sig, frame):
    print("\nInterruzione rilevata. Terminazione dei thread in corso...")
    for thread in threads:
        if thread.is_alive():
            thread.join()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def copy_file(sftp, local_file_path, remote_file_path):
    try:
        sftp.put(local_file_path, remote_file_path)
    except Exception as e:
        print(f"Errore durante la copia di {local_file_path}: {e}")


def copy_files_to_remote(base_local_dir, base_remote_dir, hostname, username, password):
    ssh = paramiko.client.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)

    sftp = ssh.open_sftp()

    directories = [
        d
        for d in os.listdir(base_local_dir)
        if os.path.isdir(os.path.join(base_local_dir, d).replace("\\", "/"))
    ]

    for i, directory in enumerate(directories):
        remote_directory = os.path.join(base_remote_dir, directory).replace("\\", "/")
        local_directory = os.path.join(base_local_dir, directory).replace("\\", "/")

        try:
            sftp.stat(remote_directory)
        except FileNotFoundError:
            sftp.mkdir(remote_directory)

        try:
            remote_files = sftp.listdir(remote_directory)
        except FileNotFoundError:
            remote_files = []

        local_files = [
            f
            for f in os.listdir(local_directory)
            if os.path.isfile(os.path.join(local_directory, f).replace("\\", "/"))
            and f.endswith(".jpg")
        ]

        if len(remote_files) < len(local_files):
            video_files = defaultdict(list)
            for file in local_files:
                video_name = "_".join(file.split("_")[:-2])
                video_files[video_name].append(file)

            print(
                "Copying directory n. "
                + str(i)
                + "/"
                + str(len(directories) - 1)
                + " ("
                + directory
                + ")"
            )

            for idx, (video_name, video_files_list) in enumerate(video_files.items()):
                thread = threading.Thread(
                    target=copy_video_files,
                    args=(
                        sftp,
                        video_name,
                        local_directory,
                        remote_directory,
                        video_files_list,
                        idx,
                    ),
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        else:
            print(
                "Files already present in directory n. "
                + str(i)
                + "/"
                + str(len(directories) - 1)
                + " ("
                + directory
                + ")"
            )

    sftp.close()
    ssh.close()


def copy_video_files(
    sftp, video_name, local_directory, remote_directory, files, position
):
    for file in tqdm(files, desc=video_name, leave=False, position=position):
        local_file_path = os.path.join(local_directory, file).replace("\\", "/")
        remote_file_path = os.path.join(remote_directory, file).replace("\\", "/")
        copy_file(sftp, local_file_path, remote_file_path)


hostname = ""
username = ""
password = ""
base_local_dir = "D:/data/ego_recordings"
base_remote_dir = ""

copy_files_to_remote(base_local_dir, base_remote_dir, hostname, username, password)
