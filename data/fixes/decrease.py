import os
import ffmpeg


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
    with open(
        output_directory + "/" + no_proper_videos_file_path, mode="a"
    ) as no_proper_videos:
        for i, directory in enumerate(directories):
            if i not in skip_directories:
                target_dir = os.path.join(output_directory, directory).replace(
                    "\\", "/"
                )
                if directory not in os.listdir(output_directory):
                    os.makedirs(target_dir, exist_ok=True)
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
                        "\nCopying directory n. "
                        + str(i)
                        + "/"
                        + str(len(directories))
                        + " ("
                        + target_dir
                        + ")"
                    )
                    # Iterate over the files
                    for file in files:
                        # Construct the input and output paths
                        input_path = os.path.join(
                            input_directory, directory, file
                        ).replace("\\", "/")
                        output_path = os.path.join(
                            output_directory, directory, file
                        ).replace("\\", "/")

                        print("Processing", input_path)
                        # Decrease the resolution and the fps using ffmpeg-python
                        (
                            ffmpeg.input(input_path)
                            .filter("scale", width=640, height=360)
                            .filter("fps", fps=15, round="up")
                            .output(output_path, vcodec="h264_nvenc", loglevel="quiet")
                            .run()
                        )


# Example usage
decrease_fps_and_send("C:/tmp", "D:/data/fixed_recordings", "no_proper_viedos.txt")
print("Done")
