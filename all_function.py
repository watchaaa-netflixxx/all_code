# pip install moviepy
# pip install mediapipe
import os
import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
import argparse
import moviepy.editor
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

####################################################################################################################    전처리 함수


# 저장되는 비디오 이름 "15s.mp4"
def vid_cut(uploaded_video, Vid_Folder_path):       # 작동 완료
    
    # input 예시
    # vid_cut('./Vid_Folder/stu1_43.mp4', './Vid_Folder/')

    video_duration = VideoFileClip(uploaded_video).duration

    # Define the clip duration (16 seconds)
    clip_duration = 16

    # Calculate the start time for the clip
    start_time = video_duration / 2 - (clip_duration / 2)

    # Extract the 16-second clip
    output_path = Vid_Folder_path + "15s.mp4"  # Output file name
    ffmpeg_extract_subclip(uploaded_video, start_time, start_time + clip_duration, targetname=output_path)

# 저장되는 이미지 이름 "frame-00" ~ "frame-31"
def vid2img(cut_video, image_Folder_path , rate=0.5, frameName='frame'):     # 작동 완료
    
    # input 예시
    # vid2img('./Time_Vid_Folder/15s.mp4', './32_imgs')  

    vidcap = cv2.VideoCapture(cut_video)
    clip = moviepy.editor.VideoFileClip(cut_video)    # 동영상 파일을 불러오기
    seconds = clip.duration                     # 동영상 전체 길이

    count = 0
    frame = 0

    if not os.path.isdir(image_Folder_path ):
        os.mkdir(image_Folder_path )

    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,frame*1000)
        success,image = vidcap.read()

        ## Stop when last frame is identified
        # print(frame)
        if frame > seconds or not success:
            break

        # print('extracting frame ' + frameName + '-%s.png' % str(count).zfill(2))
        name = image_Folder_path  + '/' + frameName + '-%s.png' % str(count).zfill(2) # save frame as PNG file
        cv2.imwrite(name, image)
        frame += rate
        count += 1

# img2data() 함수에서 사용
def resize(image, DESIRED_WIDTH, DESIRED_HEIGHT):       # 작동 완료

  # 입력 이미지의 높이와 너비를 가져옴
  h, w = image.shape[:2]

  # 이미지의 높이가 너비보다 작을 경우
  if h < w:
    # 이미지의 너비를 'DESIRED_WIDTH로 조절하고, 높이는 비율에 따라 조절
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  # 아닌 경우
  else:
    # 이미지의 높이를 'DESIRED_WIDTH'로 조절하고, 너비는 비율에 따라 조절
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

# 33개 landmark x,y,z 좌표값 csv 파일 저장 (coordinate.csv)
def img2data(image_Folder_s, data_Folder_path):         # 작동완료
   
    # img_div_path = '../32_imgs/'
    # data_div_path = '../data/'

    path = image_Folder_s
    os.chdir(path)
    uploaded = os.listdir(path)
    uploaded.sort()

    # 이미지의 원하는 높이와 너비 (480*480)
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    # Read images with OpenCV.
    images = {name: cv2.imread(name) for name in uploaded}

    # resize 함수를 통해 크기 조절
    for name, image in images.items():
      resize(image, DESIRED_WIDTH, DESIRED_HEIGHT)

    mp_pose = mp.solutions.pose # 포즈 추정 모델을 사용할 수 있게 하는 인터페이스 제공, 포즈 감지 & 랜드마크 탐색
    mp_drawing = mp.solutions.drawing_utils # 포즈 추정 결과를 시각적으로 표시
    mp_drawing_styles = mp.solutions.drawing_styles

    # 결과값의 랜드마크 이름 목록
    landmark_names = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
        "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
        "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
        "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
        "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]

    # columns 명 설정
    columns = []
    for landmark_name in landmark_names:
        columns.extend([f"{landmark_name}.x", f"{landmark_name}.y", f"{landmark_name}.z"])

    df = pd.DataFrame(columns=columns)

    #  MediaPipe Pose 모델 초기화, Pose 모델 객체 생성
    with mp_pose.Pose(
        # 정적 이미지 처리, 최소 검출 신뢰도 0.5, 모델의 복잡성 중간 수준
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:

      # images라는 딕셔너리에서 각각의 이미지와 해당 이미지의 이름 가져옴
      for name, image in images.items():
        # 이미지의 색공간을 BGR에서 RGB로 변환, pose.process() 함수를 사용하여 Pose모델에 이미지를 전달하여 포즈 추정
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 결과값을 저장할 리스트 초기화
        landmark_data = []

        # 각 랜드마크의 좌표값을 landmark_data 리스트에 추가
        for landmark in results.pose_world_landmarks.landmark:
            landmark_data.extend([landmark.x, landmark.y, landmark.z])

        # 데이터프레임에 행 추가
        df = pd.concat([df, pd.DataFrame([landmark_data], columns=columns)], axis = 0)

        output_path = data_Folder_path + 'coordinate.csv'
        df.to_csv(output_path, index=False)

# data2angle_model 함수에서 사용
def calculate_3d(point1, point2, point3):
    # Calculate the vectors between the left hand, elbow, and shoulder landmarks
    point_1_2 = [(point2[i] - point1[i]) for i in range(3)]
    point_2_3 = [(point2[i] - point3[i]) for i in range(3)]

    # Calculate the dot product and the magnitudes of the vectors
    dot_product = sum([point_1_2[i] * point_2_3[i] for i in range(3)])
    point_1_2_mag = math.sqrt(sum([coord**2 for coord in point_1_2]))
    point_2_3_mag = math.sqrt(sum([coord**2 for coord in point_2_3]))

    # Calculate the angle between the left hand, elbow, and shoulder landmarks in degrees
    angle = math.degrees(math.acos(dot_product / (point_1_2_mag * point_2_3_mag)))

    return angle


####################################################################################################################    카운트 관련


# 임의의 관절 각도 계산
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# 각 landmark 좌표 추출
def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

def detection_body_parts(landmarks):
    body_parts = pd.DataFrame(columns=["body_part", "x", "y"])

    for i, lndmrk in enumerate(mp_pose.PoseLandmark):
        lndmrk = str(lndmrk).split(".")[1]
        cord = detection_body_part(landmarks, lndmrk)
        body_parts.loc[i] = lndmrk, cord[0], cord[1]

    return body_parts

def score_table(exercise, frame , counter, status):
    cv2.putText(frame, "Activity : " + exercise.replace("_", " "),
                (600, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "Counter : " + str(counter), (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Status : " + str(status), (600, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

def score_table_plus(exercise, frame , counter, good, bad, status):
    cv2.putText(frame, "Activity : " + exercise.replace("_", " "),
                (570, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "Counter : " + str(counter), (570, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Good : " + str(good) + "  Bad : " + str(bad), (570, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Status : " + str(status), (570, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

# 각 관절마다의 각도 계산 함수 정의
class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_left_arm(self):
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        l_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        return calculate_angle(l_shoulder, l_elbow, l_wrist)

    def angle_of_the_right_arm(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        r_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        return calculate_angle(r_shoulder, r_elbow, r_wrist)

    def angle_of_the_left_leg(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        l_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
        return calculate_angle(l_hip, l_knee, l_ankle)

    def angle_of_the_right_leg(self):
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        r_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
        return calculate_angle(r_hip, r_knee, r_ankle)

    def angle_of_the_neck(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        r_mouth = detection_body_part(self.landmarks, "MOUTH_RIGHT")
        l_mouth = detection_body_part(self.landmarks, "MOUTH_LEFT")
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")

        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]
        mouth_avg = [(r_mouth[0] + l_mouth[0]) / 2,
                     (r_mouth[1] + l_mouth[1]) / 2]
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        return abs(180 - calculate_angle(mouth_avg, shoulder_avg, hip_avg))

    def angle_of_the_abdomen(self):
        # calculate angle of the avg shoulder
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]

        # calculate angle of the avg hip
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        # calculate angle of the avg knee
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        knee_avg = [(r_knee[0] + l_knee[0]) / 2, (r_knee[1] + l_knee[1]) / 2]

        return calculate_angle(shoulder_avg, hip_avg, knee_avg)

# 운동 별 카운트 기준 정의
class TypeOfExercise(BodyPartAngle):

    def __init__(self, landmarks):
        super().__init__(landmarks)

    # 시작자세 수축 -> 카운트 올라가는 시점 True
    def push_up(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        if status:
            if avg_arm_angle < 100:
                status = False
        else:
            if avg_arm_angle > 160:
                counter += 1
                status = True

        return [counter, status]

    # 시작자세 이완 -> 카운트 올라가는 시점 True
    def barbell_low(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        if status:
            if avg_arm_angle < 120:
                status = False
        else:
            if avg_arm_angle > 160:
                counter += 1
                status = True

        return [counter, status]

    # 시작자세 이완 -> 카운트 올라가는 시점 True
    def overhead_press(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        if status:
            if avg_arm_angle > 160:
                status = False
        else:
            if avg_arm_angle < 100:
                counter += 1
                status = True

        return [counter, status]

    # 시작자세 이완 -> 카운트 올라가는 시점 True
    # 서서 시작하여 바벨 잡으려 숙임 & 시작부터 바벨 잡고 있는것 경우의 수 나눠야함 (일단 시작부터 잡고 있는 걸로 짜놓음)********** test 용으로 부적합
    def dead_lift(self, counter, status):
        angle = self.angle_of_the_abdomen()
        if status:
            if angle > 160:
                status = False
        else:
            if angle < 120:
                counter += 1
                status = True

        return [counter, status]

    # 시작자세 수축 -> 카운트 올라가는 시점 True
    def squat(self, counter, status):
        left_leg_angle = self.angle_of_the_left_leg()
        right_leg_angle = self.angle_of_the_right_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2
        if status:
            if avg_leg_angle < 100:
                status = False
        else:
            if avg_leg_angle > 160:
                counter += 1
                status = True

        return [counter, status]

    def calculate_exercise(self, exercise_type, counter, status):
        if exercise_type == "push_up":
            counter, status = TypeOfExercise(self.landmarks).push_up(
                counter, status)
        elif exercise_type == "overhead_press":
            counter, status = TypeOfExercise(self.landmarks).overhead_press(
                counter, status)
        elif exercise_type == "barbell_low":
            counter, status = TypeOfExercise(self.landmarks).barbell_low(
                counter, status)
        elif exercise_type == "dead_lift":
            counter, status = TypeOfExercise(self.landmarks).dead_lift(
                counter, status)
        elif exercise_type == "squat":
            counter, status = TypeOfExercise(self.landmarks).squat(
                counter, status)

        return [counter, status]

# 영상 카운트 기준 자르는 함수
def save_video_segment(video_path, start_frame, end_frame, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 영상의 FPS 및 프레임 크기 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 영상을 읽어서 저장
    for i in range(start_frame, end_frame):
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 저장
        out.write(frame)

    # VideoWriter 및 VideoCapture 객체 해제
    out.release()
    cap.release()


    # exercise_type 설정  ************************** 나중에 분류 모델에서 출력값 받아야 함

# 카운트 기준 영상 자르기
# count_{i} 로 저장됨
def vid2time(class_int, cut_video, cut_Vid_Folder_path):        # 작동 완료

    # class_int = 정수값 (data2angle_classmodel 함수 반환값)
    # cut_video = './15s_cut_video/15s.mp4'
    # 잘라진 동영상 저장 경로 (이 폴더 내에 count_{count_i}.mp4 형식으로 저장됨)
    # cut_Vid_Folder_path = './cut_Vid_Folder/'

    # class_int로부터 exercise_type 결정하기
    # '0':'바벨 데드리프트', '1':'바벨 로우', '2':'바벨 스쿼트', '3':'오버 헤드 프레스', '4':'푸시업'

    if class_int == 0:
        exercise_type = "dead_lift"
    elif class_int == 1:
        exercise_type = "barbell_low"
    elif class_int == 2:
        exercise_type = " squat"
    elif class_int == 3:
        exercise_type = "overhead_press"
    elif class_int == 4:
        exercise_type = "push_up"

    # 비디오 업로드
    cap = cv2.VideoCapture(cut_video)

    # status 변화를 저장할 리스트
    status_list = []

    # 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # setup mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        prev_status = True    # status 바뀔 때 동영상 시간 초 표시 위함
        prev_frame_time = 0

        counter = 0  # movement of exercise
        status = True  # state of move

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # frame setup
            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status)

                prev_status = status

            except Exception as e:
                print(f'Error in row: {e}')

            # status 변화를 저장
            status_list.append(status)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # 카운트 별로 동영상 자르고 저장
        start_frame = 1
        count_i = 1

        # 각각의 status 변화를 확인하며 동영상 저장 (모든 운동에 대해 적용됨!)
        # False -> True 기준으로 나눔
        for i in range(3, len(status_list)):
            if status_list[i-1] == False and status_list[i] == True:
                end_frame = int(i * (total_frames / len(status_list))) + 25
                # 저장할 파일 경로 설정
                output_path = cut_Vid_Folder_path + f'count_{count_i}.mp4'
                # 동영상 저장
                save_video_segment(cut_video, start_frame, end_frame, output_path)
                start_frame = int(i * (total_frames / len(status_list)))
                count_i += 1

        cap.release()
        cv2.destroyAllWindows()


####################################################################################################################    class model


# return = int(y_pred_class[0]) (운동 분류 값)
# class = classmodel(...)
def class_model(uploaded_video, Vid_Folder_path, image_Folder_path, data_Folder_path , model_Folder_path):          # 작동 모름

    # input 예시
    # uploded_video = '../uploded_video/stu1_43.mp4'    (사용자가 업로드한 영상)
    # Vid_Folder_path = '../Vid_Folder/'	            (15초 잘라진 동영상이 있는 폴더)
    # image_Folder_path = '../image_Folder'	            (32개 이미지가 있는 폴더)
    # data_Folder_path = '../data/'		                (이미지에서 좌표 뽑아낸 csv가 있는 폴더)
    # model_Folder_path = '../model/'	                (classifycation model이 있는 폴더)


    # 전처리 함수 실행
    #####################################################
    # 15초 cut하여 저장
    vid_cut(uploaded_video, Vid_Folder_path)

    # 32개 이미지 cut하여 저장
    cut_video = Vid_Folder_path + '15s.mp4'
    vid2img(cut_video , image_Folder_path ) 

    # 32개 이미지에서 좌표값 뽑아내어 csv 파일 저장
    image_Folder_s = image_Folder_path +'/'
    img2data(image_Folder_s, data_Folder_path )
    #####################################################

    data_file = data_Folder_path + 'coordinate.csv'
    class_model_file = model_Folder_path +'classify_model.h5'

    # 분류 model load
    class_model = tf.keras.models.load_model(class_model_file)

    # 32*99 dataframe
    df_cor = pd.read_csv(data_file)

    point_groups = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
    ('LEFT_ANKLE', 'LEFT_KNEE', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER'),
    ('RIGHT_ANKLE', 'RIGHT_KNEE', 'RIGHT_HIP')
    ]

    # 열 이름 리스트 만들기
    column_names = [f'{group[0]}_ANGLE' for group in point_groups]

    # 각 그룹의 각도 계산 및 데이터프레임에 추가
    for group, col_name in zip(point_groups, column_names):
        df_cor[col_name] = df_cor.apply(lambda row: calculate_3d(
            [row[f'{group[0]}.x'], row[f'{group[0]}.y'], row[f'{group[0]}.z']],
            [row[f'{group[1]}.x'], row[f'{group[1]}.y'], row[f'{group[1]}.z']],
            [row[f'{group[2]}.x'], row[f'{group[2]}.y'], row[f'{group[2]}.z']]
        ), axis=1)

    df_cor.drop(columns=['img_key'], inplace = True)

    X = df_cor

    sequence_length = 32  # 시퀀스 길이 설정
    Xsequence = []

    for i in range(0, len(X) - sequence_length + 1, sequence_length):
        Xsequence.append(X[i:i+sequence_length])

    Xsequence = np.array(Xsequence)

    y_pred = class_model.predict(Xsequence)
    
    y_pred_class = np.argmax(y_pred, axis=1)

    return int(y_pred_class[0])


####################################################################################################################    correct model


def correct_model(class_int, Vid_Folder_path, cut_Vid_Folder_path):

    cut_video = Vid_Folder_path + '15s.mp4'
    vid2time(class_int, cut_video, cut_Vid_Folder_path)


####################################################################################################################    skelton 및 운동정보 화면 저장

test_label = [1, 0, 1, 1, 0]

def vid2Mvid(class_int, Vid_Folder_path, MVid_Folder_path):

    # class_int = 0~4                       (class_model 함수 반환값)
    # Vid_Folder_path = '../Vid_Folder/'    (15초 영상 있는 폴더)
    # MVid_Folder_path = '../MVid_Folder/'  (최종 결과 영상 있는 폴더)

    if class_int == 0:
        exercise_type = "dead_lift"
    elif class_int == 1:
        exercise_type = "barbell_low"
    elif class_int == 2:
        exercise_type = " squat"
    elif class_int == 3:
        exercise_type = "overhead_press"
    elif class_int == 4:
        exercise_type = "push_up"

    # 15초 video 파일
    cut_video = Vid_Folder_path + '15s.mp4'

    # skeleton only video 파일 저장하기
    skeleton_output_path = MVid_Folder_path + 'MVid.mp4'

    cap = cv2.VideoCapture(cut_video)

    # 빈 프레임 규격 설정 (흰 배경)
    frame_width = 800
    frame_height = 480

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(skeleton_output_path, fourcc, 30, (frame_width, frame_height))

    # status 변화를 저장할 리스트
    status_list = []

    # 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # setup mediapipe
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        prev_status = True    # status 바뀔 때 동영상 시간 초 표시 위함
        prev_frame_time = 0

        counter = 0  # movement of exercise
        status = True  # state of move

        # 정확도 측정용 카운트
        good = 0
        bad = 0
        acc_i = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # 불러온 동영상 frame setting
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # mediapipe 적용
            results = pose.process(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                # count 계산
                landmarks = results.pose_landmarks.landmark
                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status)

                # 정확도 label 1이면 good 이라고 설정
                if prev_status == False and status == True:
                    if test_label[acc_i] == 1:
                        good += 1
                        acc_i += 1
                    else:
                        bad += 1
                        acc_i += 1

                prev_status = status

            except Exception as e:
                print(f'Error in row: {e}')

            # status 변화를 저장
            status_list.append(status)

            # 빈 프레임 생성 (흰 배경)
            blank_frame = np.ones((frame_height, frame_width, 3), np.uint8) * 255

            # RGB로 변환 (없으면 skeleton 품질 안좋아짐)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose 적용
            results = pose.process(frame_rgb)

            if results.pose_landmarks is not None:

                # 화면에 정보 표시
                blank_frame = score_table_plus(exercise_type, blank_frame, counter, good, bad, status)

                # 스켈레톤 그리기
                mp_drawing.draw_landmarks(
                    blank_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # output 영상에 추가
                out.write(blank_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()



