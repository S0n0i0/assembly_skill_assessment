import os
import ffmpeg
import paramiko
import paramiko.client


def decrease_fps_and_send(
    input_directory, output_directory, no_proper_videos_file_path
):
    # Get the directories within the given directory path
    directories = [
        d
        for d in os.listdir(input_directory)
        if os.path.isdir(os.path.join(input_directory, d))
    ]

    skip_directories = []

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(
        "marzola.disi.unitn.it",
        username="simone.compri",
        password="Simone@2000",
    )
    sftp = client.open_sftp()
    print(output_directory + "/" + no_proper_videos_file_path)
    with sftp.file(
        output_directory + "/" + no_proper_videos_file_path, mode="a", bufsize=1
    ) as no_proper_videos:
        for i, directory in enumerate(directories):
            if i not in skip_directories:
                target_dir = os.path.join(output_directory, directory).replace(
                    "\\", "/"
                )
                if directory not in sftp.listdir(output_directory):
                    sftp.mkdir(target_dir)
                    exist_target = False
                else:
                    exist_target = True

                # Get the files within each directory
                files = [
                    f
                    for f in os.listdir(os.path.join(input_directory, directory))
                    if os.path.isfile(os.path.join(input_directory, directory, f))
                ]

                if exist_target:
                    present_files = sftp.listdir(target_dir)
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

                if len(files) == 0 and (not exist_target or len(present_files) == 0):
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
                    print()
                    # Iterate over the files
                    # tmp_input_directory = "D" + input_directory[1:]
                    for file in files:
                        # Construct the input and output paths
                        """input_path = os.path.join(
                            input_directory, directory, file
                        ).replace("\\", "/")"""
                        output_path = os.path.join(
                            input_directory, directory, file  # f"decreased_{file}"
                        ).replace("\\", "/")

                        # Decrease the fps using ffmpeg-python
                        # ffmpeg.input(input_path).output(output_path, r=15).run()
                        """(
                            ffmpeg.input(input_path)
                            .filter("fps", fps=15, round="up")
                            .output(output_path)
                            .run()
                        )"""

                        # Send the file to the SSH server
                        print(
                            "-",
                            output_path,
                            "->",
                            f"{output_directory}/{directory}/{file}",
                        )
                        sftp.put(output_path, f"{output_directory}/{directory}/{file}")

                        os.remove(output_path)

    sftp.close()
    client.close()


# Example usage
decrease_fps_and_send(
    "D:/data/fixed_recordings",
    "/home/simone.compri/data/fixed_recordings",
    "no_proper_viedos.txt",
)
print("Done")