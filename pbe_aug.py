import os
import cv2
import random
import sys
import numpy as np
# Add the subdirectory to the Python path

# Import the main_script module from the subdirectory
from scripts import inference
from tqdm import tqdm
import argparse

#takes bbox coords => returns yolo style bbox coords 
#yolo 포맷으로 bbox 좌표 변환
def bbox2yolo(xmin, ymin, xmax, ymax, img_shape):
    b_center_x = (xmin + xmax) / 2 
    b_center_y = (ymin + ymax) / 2
    b_width    = (xmax - xmin)
    b_height   = (ymax - ymin)
    image_h, image_w, image_c = img_shape
    b_center_x /= image_w 
    b_center_y /= image_h 
    b_width    /= image_w 
    b_height   /= image_h 
    return b_center_x, b_center_y, b_width, b_height


#randomly crops img by 1/rows*cols size => returns cropped img, cropped coordinates, random selceted # of row
def crop_rand_region(img, rows, cols):
    print('crop rand region called')
    random.seed()
    original_image = img
    height, width, _ = original_image.shape
    cropped_width = width // cols
    cropped_height = height // rows
    row = random.randrange(rows)
    col = random.randrange(cols)

    left = col * cropped_width
    upper = row * cropped_height
    right = left + cropped_width
    lower = upper + cropped_height
    
    cropped_image = original_image[upper:lower, left:right]
    return cropped_image, [upper,lower, left, right], row

def pbe_aug():

    # test 데이터셋 동영상 파일명 list
    lst = ['A-15-1-(12).mp4', 'A-15-2-(51).mp4', 'A-12-1-(10).mp4', 'A-12-2-(11).mp4', 'A-12-2-(64).mp4', 'A-15-3-12.mp4', 'A-15-3-20.mp4', 'A-12-2-(102).mp4', 'B-sunrise-29.mp4', 'A-09-(25).mp4', 'A-12-2-(53).mp4', 'B-09-6.mp4', 'A-sunrise-28.mp4', 'B-15-19.mp4', 'A-12-2-(30).mp4', 'A-12-1-(18).mp4', 'A-15-3-29.mp4', 'A-15-2-(77).mp4', 'A-12-3-(10).mp4', 'A-12-1-(5).mp4', 'A-sunrise-11.mp4', 'A-sunrise-19.mp4', 'A-09-(23).mp4', 'B-12-1.mp4', 'A-12-1-(40).mp4', 'A-12-3-(19).mp4', 'A-15-1-(29).mp4', 'A-12-3-(62).mp4', 'A-09-(19).mp4', 'B-09-28.mp4', 'B-15-29.mp4', 'B-09-12.mp4', 'B-sunrise-18.mp4', 'A-09-(61).mp4', 'A-09-(48).mp4', 'A-09-(45).mp4', 'B-12-29.mp4', 'A-15-2-(2).mp4', 'A-09-(33).mp4', 'A-15-2-(31).mp4', 'A-12-1-(7).mp4', 'A-12-3-(1).mp4', 'A-12-3-(37).mp4', 'A-09-(20).mp4', 'B-12-4.mp4', 'A-15-2-(100).mp4', 'A-09-(29).mp4', 'B-12-11.mp4', 'A-15-1-(18).mp4']

    #pbe

    #SD Model initialization
    model = inference.init_model_ret()
    print('model initialized!')

    #root directory of dataset
    rt = './gookbang' 
    save_dir = './gookbang/pbe_aug_res'
    count = 0
    #데이터셋 구조 따라 iterate
    for subdir in tqdm(os.listdir(rt)):
        if subdir == 'pbe_aug_res' : 
            continue
        for subsubdir in os.listdir(os.path.join(rt,subdir)):
            if not subsubdir.startswith('.'):
                for file in os.listdir(os.path.join(rt,subdir,subsubdir)):
                    if not file.startswith('.') and file not in lst and '.mp4' in file:
                        print(os.path.join(rt,subdir,subsubdir,file))
                        fname = file.strip('.mp4')
                        
                        #영상 파일 오픈
                        cap = cv2.VideoCapture(os.path.join(rt,subdir,subsubdir,file))
                        frame_count =0 #프레임 수 

                        #영상 내 프레임들 모두 iterate 하여 처리
                        while True:
                            ret_val, frame = cap.read() 
                            #4초에 프레임 한장씩 받아옴
                            if frame_count%120 != 0:
                                frame_count+=1
                                continue
                            if not ret_val:
                                break
                            #1/64 사이즈로 random crop
                            cropped_img, coords, row = crop_rand_region(frame,8,8)
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                            mask = './mask_sm.jpg'

                            #sd 모델 이nsfw outout을 return 할 경우 black image 를 반환함. 이를 대비하여 black image 일 경우 재추론
                            while True:
                                ret_img = inference.main(model, mask ,cropped_img,'./ghillie_sd_gen.png')
                                ret_img = cv2.cvtColor(np.asarray(ret_img), cv2.COLOR_RGB2BGR)

                                mean_val = ret_img.mean()
                                if mean_val<10:
                                    print('black img found.')
                                else:
                                    break  

                            #crop된 이미지 별도 저장
                            upper,lower, left, right = coords
                            #원본 frame에 paint by example 결과물 재합성
                            frame[upper:lower, left:right] = ret_img
                            frame_count+=1
                            
                            #최종 결과물 저장
                            cv2.imwrite(f'{save_dir}/{fname}_pbe_{frame_count}.jpg', frame)

                            #bbox 좌표 yolo style 로 변환 후 txt 파일로 저장
                            mask_image = cv2.imread(mask , cv2.IMREAD_GRAYSCALE)
                            mask_image = cv2.resize(mask_image, (512, 512))
                            ret_h,ret_w,c = ret_img.shape
                            mask_image = cv2.resize(mask_image, (ret_w,ret_h))  

                            white_pixels_mask = (mask_image == 255)  
                            white_pixel_indices = np.where(white_pixels_mask)
                            #mask의 흰색 영역 좌표 계산
                            min_x = np.min(white_pixel_indices[1])
                            min_y = np.min(white_pixel_indices[0])
                            max_x = np.max(white_pixel_indices[1])
                            max_y = np.max(white_pixel_indices[0])

                            #bbox 좌표 txt 파일로 저장
                            bbox = min_x+left, min_y+upper, max_x+left, max_y+upper
                            yolobbox = bbox2yolo(min_x+left, min_y+upper, max_x+left, max_y+upper, frame.shape) #yolo style로 bbox coords 변환
                            with open(f'{save_dir}/{fname}_pbe_{frame_count}.txt', 'w') as txt_result:
                                txt_result.write(f"0 {yolobbox[0]} {yolobbox[1]} {yolobbox[2]} {yolobbox[3]}")

                            print(f'imgs saved for {file} @ frame#{frame_count}. coords: {coords}, row: {row}')

                        #video source file read 종료
                        if cap.isOpened():	
                            cap.release()	

if __name__ == "__main__":
    pbe_aug()

breakpoint()
