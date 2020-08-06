#!/usr/bin/env python
import os
import sys
import cv2
import datetime

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-video_path", type=str, required=True)
    parser.add_argument("-output_video_path", type=str, default="result.MP4")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())

    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        raise Exception("failed to open {}".format(args.video_path))

    width = int(capture.get(3))
    height = int(capture.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fps = 30.0
    out_video = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

    _total_ms = 0
    count_frame = 0
    while capture.isOpened():
        ret, frame = capture.read()
        count_frame += 1

        if not ret:
            break

        start = datetime.datetime.now()
        _, overlay = segmentation_handler.run(frame, only_mask=False)
        _total_ms += (datetime.datetime.now() - start).total_seconds() * 1000

        cv2.imshow("result", overlay)
        out_video.write(overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("processing time one frame {}[ms]".format(_total_ms / count_frame))

    capture.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
