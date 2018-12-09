from LaneFind import *
from os.path import join, basename
from collections import deque

resize_h, resize_w = 540, 960


# test on images
images_dir = 'test_images'
test_images = [join(images_dir, name) for name in os.listdir(images_dir)]

for img in test_images:

    print('Processing image: {}'.format(img))


    out_path = join('output_images', basename(img))
    in_image = cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    out_image = pipeline([in_image])
    cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))


# test on video
video_dir = 'test_videos'
test_video = [join(video_dir, name) for name in os.listdir(video_dir)]

for vdeo in test_video:
    print('Processing video: {}'.format(vdeo))
    cap = cv2.VideoCapture(vdeo)
    out = cv2.VideoWriter(join('test_videos_output', basename(vdeo)),
                                    fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                                    fps=20.0, frameSize=(resize_w, resize_h))

    frame_buffer = deque(maxlen=10)
    while cap.isOpened():
        ret, color_frame = cap.read()
        if ret:
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            color_frame = cv2.resize(color_frame, (resize_w, resize_h))
            frame_buffer.append(color_frame)
            blend_frame = pipeline(frames=frame_buffer)
            out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
            cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
