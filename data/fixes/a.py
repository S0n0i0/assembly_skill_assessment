import cv2

# open a video
video = ":/data/ego_recordings/nusar-2021_action_both_9015-c03a_9015_user_id_2021-02-02_163503/HMC_21176623_mono10bit.mp4"
cap1 = cv2.VideoCapture(f"D{video}")
cap2 = cv2.VideoCapture(f"E{video}")

# get the frames per second
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)

print(f"fps1: {fps1}, fps2: {fps2}")

# get the total number of frames

total_frames1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
total_frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"total_frames1: {total_frames1}, total_frames2: {total_frames2}")

resolution1 = (cap1.get(cv2.CAP_PROP_FRAME_WIDTH), cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution2 = (cap2.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"resolution1: {resolution1}, resolution2: {resolution2}")

# set the video to the frame number
frame = 11516
actual_frame = frame / 4
try:
    cap1.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame)

    # read the frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # display the frame
    if ret1:
        cv2.imshow(f"frame1: {actual_frame}", frame1)
    else:
        print("frame1 not read")
    if ret2:
        cv2.imshow(f"frame2: {frame}", frame2)
    else:
        print("frame2 not read")
    if ret1 or ret2:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
except Exception as e:
    print(e)
