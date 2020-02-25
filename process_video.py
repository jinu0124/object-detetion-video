import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names
import time
from datetime import datetime
import math
from skimage.transform import resize  # image resizing을 위해

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
import ffmpeg

#capture = cv2.VideoCapture(1)
capture = cv2.VideoCapture("road_960_512.mp4") # cv의 VideoCapture 클래스를 사용
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
Video_w = (capture.get(cv2.CAP_PROP_FRAME_WIDTH))
Video_h = (capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
print("fps :", fps)
length = frame/fps

print("Width :", Video_w, "Height :", Video_h)

Video_w_20 = round(Video_w * 0.2)#반올림 함수 round
Video_w_80 = round(Video_w - Video_w_20)
Video_h_35 = round(Video_h * 0.35)

print(Video_w_20, Video_w_80, Video_h_35)
#Video_w_20 : 화면 상 좌 20% 지점 / Video_w_80 : 화면 상 우 80% 지점

codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked_road_20%35%_2x_50_inc.avi', codec, 30.0, size)

flag = 0
masking = 0 # 인코딩(Boxing) 처리 성능 향상을 위해서 한프레임씩 건너서 Boxing(Object Detecting) -> 속도 향상
print("Start masking")
now = datetime.now()
print("Start at :", now)
start = round(time.time())
#
while(1):#capture.isOpened()
    ret, frame = capture.read() # ret 받은 이미지가 있는지 여부 , 각 프레임 받기

    if ret and masking == 0:
        results = model.detect([frame], verbose=0)# 모델 사용 -> 모델에서 Forward Compute 해서 Detection 결과를 반환
        r = results[0]

      #  print("visualize_cv2 LINE 131 :", r)
      #  print("class names :", class_names)
        #{'rois': array([[1061, 11, 1280,  201],

        masking = masking + 1
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], Video_w, Video_w_20, Video_w_80, Video_h_35
        )
        # display_instances를 호출(수행)할 때 마다
        output.write(frame)
        cv2.imshow('frame', frame)#원본 영상에 Masking이 입혀진 영상 보여주기 함수

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    elif ret and masking > 0:
        masking = masking + 1
        if masking == 2: # 몇 프레임 당 Compute할것인지
            masking = 0
        if cv2.waitKey(1) & 0xFF == ord('q'): # waitkey & 0xFF = 1111 1111 == 'q'
            break
        output.write(frame) # Model forward Compute를 거치지 않고 바로 출력
        cv2.imshow('Drive', frame)
    else:
        break

now = datetime.now()
print("End at :", now)

end = round(time.time())
taken_time = end - start
minute = math.floor(taken_time/60)
sec = taken_time%60
print("taken_time :", minute, ":", sec)

rate = length/taken_time
print("encoding rate :", rate, ": 1", "1보다 커야 실시간O")
capture.release()
output.release()
cv2.destroyAllWindows()
