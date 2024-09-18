import os
import paramiko
import paramiko.client
import stat


def download_videos(local_directory, remote_directory, no_proper_videos_file_path):
    skip_directories = []

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(
        "marzola.disi.unitn.it",
        username="simone.compri",
        password="Simone@2000",
    )
    sftp = client.open_sftp()
    with sftp.file(
        remote_directory + "/" + no_proper_videos_file_path, mode="a", bufsize=1
    ) as no_proper_videos:
        # Get the directories within the given directory path
        directories = sftp.listdir(remote_directory)
        for i, directory in enumerate(directories):
            # check if directory is a valid directory
            if i not in skip_directories:
                try:
                    target_dir = os.path.join(local_directory, directory).replace(
                        "\\", "/"
                    )
                    if os.path.isdir(os.path.join(local_directory, directory)):
                        exist_target = True
                    else:
                        os.makedirs(target_dir, exist_ok=True)
                        exist_target = False

                    # Get the files within each remote directory
                    files = [
                        f
                        for f in sftp.listdir(
                            os.path.join(remote_directory, directory).replace("\\", "/")
                        )
                        if f.endswith(".mp4")
                    ]

                    if exist_target:
                        present_files = os.listdir(target_dir)
                        print(
                            "Files already present in directory n. "
                            + str(i)
                            + "/"
                            + str(len(directories))
                            + " ("
                            + target_dir
                            + "):",
                            present_files,
                        )
                        files = [f for f in files if f not in present_files]

                    if len(files) == 0 and (
                        not exist_target or len(present_files) == 0
                    ):
                        print(
                            "The directory n. "
                            + str(i)
                            + " ("
                            + target_dir
                            + ") has no proper videos"
                        )
                        no_proper_videos.write(target_dir + "\n")
                        no_proper_videos.flush()
                    elif len(files) > 0:
                        print(
                            "Copying directory n. "
                            + str(i)
                            + "/"
                            + str(len(directories))
                            + " ("
                            + target_dir
                            + ")"
                        )
                        # Iterate over the files
                        for file in files:
                            print("Processing file: " + file)
                            # Download the file
                            sftp.get(
                                os.path.join(remote_directory, directory, file).replace(
                                    "\\", "/"
                                ),
                                os.path.join(target_dir, file),
                            )
                            print("File processed:", os.path.join(target_dir, file))
                except Exception as e:
                    print("Error processing element n. " + str(i) + ":", e)

    sftp.close()
    client.close()


# Example usage
download_videos(
    "D:/data/recordings", "/home/simone.compri/recordings", "no_proper_viedos.txt"
)
print("Done")
