import cv2
import numpy as np
import os
import sys
import coco
import utils
import model as modellib
import time
#from samples.mouse import mouse
from samples.load import load
import math
# import 바꿔주기

#jump over to process_video.py
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_line_0010_multi.h5")#트레이닝된 모델로 부터 Weight를 가져오기 기존 COCO
if not os.path.exists(COCO_MODEL_PATH):
    print("Downloading COCO_MODEL_PATH")
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(load.loadConfig): #coco.CocoConfig -> Coco 모델이 아닌 다른 모델 사용시, Config의 NUM_Class
    #가 변경되어야 하므로 import custom.py 해서 config 내용을 가져온다.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #영상에서는 한번에 1frame => 1이미지씩 inference(추론,계산) 할것이기에 IMAGE_PER_GPU = 1

config = InferenceConfig()
config.display()

#Create model in inference MODE
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
#Load weights from COCO_MODEL_PATH.h5
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = [
'BG', 'car', 'Slane', 'Llane'
]


# BG, person, bicycle, car, ~~
'''
class_names = [
    'BG', 'load', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]
'''

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)#zip 함수는 두 개의 인자를 쌍(2개씩)으로 새롭게 묶어줌 -> [1,3,5],[2,8,10] -> [1,2],[3,8],[5,10]
}# Class마다 서로다른 Color로 매칭 시키기 위해 사용(Boxes 색, font 색)


def apply_mask(image, mask, color, alpha=0.5):  #여기서 image는 kernel 색상 RGB
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


#L_inc, R_inc = 0.0, 0.0

def display_instances(image, boxes, masks, ids, names, scores, Video_w, Video_w_20, Video_w_80, Video_h_35): # Video_w와 Video_h를 추가로 받아옴 영상의 해상도 받기(Ex. w : 720, h : 1280)
    """
        take the image and results and apply the mask, box, and Label
    """
    #Video_h_15는 영상 화면 상 상단 15%는 안식하지 않을 것이기 때문에 설정하였음(좌상단 부터 좌표 0,0)
    n_instances = boxes.shape[0] # 분류 인식할 classes 의 갯수 (COCO data set을 사용할때)
    names_num = list() # names_num를 담기위한 list 선언
    #print("Here is display_instances Video_w :", Video_w, Video_h)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    AR_inc = 0
    AL_inc = 0
    laneR_num = 0
    laneL_num = 0
    for i in range(n_instances): # 인식된 Boxes의 갯수 -> 동 Frame에서 Detecting 순서(i)는 정확도가 높은순으로 Numbering  : (정확도최대)1, 2, 3, 4, ....

        y1, x1, y2, x2 = boxes[i]
        x = (x1 + x2) / 2  # 생성된 박스(Detection)의 좌우Center 좌표

        if not np.any(boxes[i]) or Video_w_80 < x or x < Video_w_20 or y2 < Video_h_35:
            #continue # Python의 continue : 아래 코드를실행하지않고 넘김(Pass)
            return image
        # box가 없을 시, continue를 통해서 받은 영상 이미지 그대로 반환
        #print("boxes[i]", boxes[i])
        #boxes[i] 는 각 박스의 좌표 상좌하우 순으로 리스트형태로 들어있음

        # 아래 if문은 영상에 mask를 씌우는 역할
        if scores[i] >= 0.87:  #and names[ids[i]] == 'mouse'      <- Coco Model에서 특정 객체만 Detection boxing할때 조건
          #  print("names[i] : ", names[i]) #boxes[i] 는 i번째 Detection 객체의 BOX 상좌하우 좌표값을 가짐

            a = " L"
            b = " R"
            #names_num.append(names[ids[i]] + a) #동시성 객체에 번호를 붙여서 구분
            # BOX들 중에서 정확도가 높은 순으로 네이밍 번호가 들어감 1, 2, 3, ... 순
            #print("image Location :", names_num[i], "y1(상) :",y1, "y2(하) :",y2, "x1(좌) :",x1, "x2(우) :", x2)
            #좌상단이 좌표 0, 0 입니다.
            if (x1+x2)/2 < (Video_w/2): # and video_h_15 < y1 : <- Index 에러 발생 가능(label = names_num[i]에서 네이밍이 안들어 갈 수 있음 )
                names_num.append(names[ids[i]] + a)
                L_inc = -((y2-y1)/(x2-x1)) # Left lane 박스의 기울기
                L_inc = (math.atan(L_inc) * (180/math.pi))
                #L_loc = (x2+x1)/2 # 박스의 좌표값에 따라서 Lane 유지하기 추가하기
                if L_inc < -22:
                    AL_inc = L_inc + AL_inc
                    laneL_num = laneL_num + 1
            else:
                names_num.append(names[ids[i]] + b)     #박스 R, L 에 따라서 이름 붙이기 -> 추후 활용
                R_inc = ((y2-y1)/(x2-x1))
                R_inc = (math.atan(R_inc) * (180 / math.pi))
                if R_inc > 22:
                    AR_inc = R_inc + AR_inc
                    laneR_num = laneR_num + 1 # 차선 Detection 박스에서 박스의 대각으로의 기울기를 구해서 양쪽 차선 각도에 따라서 L: - 값  , R: + 값

            #label = names[ids[i]]
            label = names_num[i]#Box에 네이밍할 네임 -> names[ids[i]]에서 names_num[i]로 바꿔서 박스별로 넘버를 매기고 표시
            color = class_dict[names[ids[i]]] # 원래 class_dict는 [label]이 인자였지만, label을 바꿨으므로 names[ids[i]]로 변경시킴
            #color 에는 위에서 정의한 class_names [ , , ]의 라벨링 명과 같아야하기 때문

            score = scores[i]
            caption = '{} {:.3f}'.format(label, score) if score else label # 소수점 3자리까지 박스 정확도 표시
            mask = masks[:, :, i]

            image = apply_mask(image, mask, color)# Mask 씌우기
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)      #cv2함수 : 박스 만들기
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )
    if AL_inc != 0 and AR_inc != 0: # 양쪽 차선 둘다 인식되었을 때  -> 추후 한쪽 Detection이 없을 때도 Steering 구현하기###
        AL_inc = AL_inc/laneL_num
        AR_inc = AR_inc/laneR_num
        inc = AL_inc + AR_inc
        #print(inc)
        if inc > 0:
            turn = "R"
        else:
            turn = "L"
        image = cv2.putText(image, '{}{:.3f}'.format(turn, inc), (Video_w_20*2, Video_h_35), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2) # 영상에 text 입히기

    return image

if __name__ == '__main__':
    """
        test everything
    """

    capture = cv2.VideoCapture(0)

    # these 2 lines can be removed if you dont have a 1080p camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:

        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)

        r = results[0]
        #print("visualize_cv2 LINE 131 :", r)
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], Video_w, Video_w_20, Video_w_80, Video_h_35
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()