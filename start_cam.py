import numpy as np
import cv2
import os
from datetime import datetime
from time import time

save_dir = "captured/"
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
delta_thresh = 5
contour_min_area = 100
save_interval = 0.2

def main():
    image_saved = 0
    frame_avg = None
    time_last = time()

    #fourcc = cv2.VideoWriter_fourcc(*'X264')
    #video_out = cv2.VideoWriter('cctv_recording.avi', fourcc, 20.0, (640,480))
    while True:
        ret, frame_ref = cap.read()
        motion_detected = False
        if ret:
            frame_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            #frame_ref = cv2.equalizeHist(frame_ref)
            frame = cv2.GaussianBlur(frame_ref, (21, 21), 0)
            #frame = cv2.fastNlMeansDenoising(frame_ref,None)
            #cv2.imshow('frame',frame)

            if frame_avg is None:
                print("[INFO] starting background model...")
                frame_avg = frame.copy().astype(np.float32)
                continue

            cv2.accumulateWeighted(frame,frame_avg,0.1)
            frame_delta = cv2.absdiff(frame, cv2.convertScaleAbs(frame_avg))

            thresh = cv2.threshold(frame_delta, delta_thresh, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < contour_min_area:
                    continue
                motion_detected = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame_ref, (x-10, y-10), (x + w+10, y + h+10), 255, 1)
                #cv2.line(frame_ref,(x,y+h),(x+w,y+h),255,1)

            if motion_detected:
                time_now = time()
                if (time_now - time_last) >= save_interval:
                    if image_saved % 5000 == 0:
                        # Separate captured images into subfolders to prevent slow disk read.
                        save_subdir = str(image_saved) + "/"
                        if not os.path.exists(save_dir + save_subdir):
                            os.makedirs(save_dir + save_subdir)
                    fname = datetime.now().strftime("%d%m%y_%H%M%S_%f")
                    cv2.imwrite(save_dir + save_subdir + fname + ".png", frame_ref)
                    image_saved += 1
                    time_last = time_now

            #video_out.write(frame_ref)
            cv2.imshow('frame_ref',frame_ref)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()

main()
