import os
import cv2


def is_video_corrupted(video_path, log=True):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if log:
                print("Errore nell'apertura del file video.")
            return True

        # Ottieni il numero totale di frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Vai all'ultimo frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # Leggi l'ultimo frame
        ret, frame = cap.read()

        if not ret:
            if log:
                print("Errore nella lettura dell'ultimo frame.")
            return True

        # Controlla se il frame è vuoto
        if frame is None:
            if log:
                print("L'ultimo frame è vuoto.")
            return True

        return False

    except cv2.error as e:
        if log:
            print(f"Errore OpenCV: {e}")
        return True
    except Exception as e:
        if log:
            print(f"Errore generico: {e}")
        return True
    finally:
        cap.release()


def check_videos(videos_directory: str, save_path: str = None):
    directories = [
        d
        for d in os.listdir(videos_directory)
        if os.path.isdir(os.path.join(videos_directory, d))
    ]
    faulty_videos = []
    count = 0

    for d in directories:
        video_directory = os.path.join(videos_directory, d)
        videos = os.listdir(video_directory)
        for v in videos:
            video_path = os.path.join(video_directory, v)
            print(f"Checking video {count}: {video_path}")
            if is_video_corrupted(video_path) and save_path is not None:
                faulty_videos.append(video_path)
                print("Video segnato come non valido")
            count += 1
    print("Done")
    if save_path is not None:
        with open(save_path, "w") as f:
            for v in faulty_videos:
                f.write(f"{v}\n")


if __name__ == "__main__":
    videos_directory = "E:/data/ego_recordings"
    log_path = "data/fauly_videos.txt"
    check_videos(videos_directory, log_path)
