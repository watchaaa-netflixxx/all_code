import mediapipe as mp
import argparse
import pandas as pd
import numpy as np
import cv2


## 카운트 준비물
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

## 카운트 기준 영상 자르기
def vid2time(class_int, video_source, cut_div_path):        # 작동 완료

    # class_int = 정수값 (data2angle_classmodel 함수 반환값)
    # video_source = './15s_cut_video/15s.mp4'
    # 잘라진 동영상 저장 경로 (이 폴더 내에 count_{count_i}.mp4 형식으로 저장됨)
    # cut_div_path = './count_cut_video/'

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
    cap = cv2.VideoCapture(video_source)

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
                output_path = cut_div_path + f'count_{count_i}.mp4'
                # 동영상 저장
                save_video_segment(video_source, start_frame, end_frame, output_path)
                start_frame = int(i * (total_frames / len(status_list)))
                count_i += 1

        cap.release()
        cv2.destroyAllWindows()


