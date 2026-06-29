import cv2
import os
import asyncio
import edge_tts
import time
#=========================================== 음성인식을 위한 코드
import speech_recognition as sr
import math # 몇 시 방향 계산에 필요
import numpy as np # 한글 라벨이 ???로 깨지는 문제 해결에 필요
from PIL import ImageFont, ImageDraw, Image

def listen_food():

    recognizer = sr.Recognizer()

    with sr.Microphone(device_index=0) as source:

        print("🎤 마이크 준비")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=0.5
        )

        print("🎤 지금 말하세요")

        audio = recognizer.listen(
            source,
            timeout=5,
            phrase_time_limit=3
        )

        print("🎤 녹음 완료")

    text = recognizer.recognize_google(
        audio,
        language="ko-KR"
    )

    return text
#===========================================
import Jetson.GPIO as GPIO

VIB_PIN = 12

GPIO.setmode(GPIO.BOARD)

GPIO.setup(
    VIB_PIN,
    GPIO.OUT
)
#==========================================
from ultralytics import YOLO

# ==========================================
# Edge TTS
# ==========================================
async def speak(text):

    communicate = edge_tts.Communicate(
        text=text,
        voice="ko-KR-SunHiNeural"
    )

    await communicate.save("voice.mp3")

    # os.system("mpg123 -q voice.mp3")
    os.system(
    "mpg123 -o alsa -a plughw:1,0 -q voice.mp3"
)


def speak_korean(text):

    try:
        asyncio.run(speak(text))

    except Exception as e:
        print("TTS 오류:", e)

# ==========================================
# 1. 기본 설정
# ==========================================
model_path = 'best.pt'
video_path = 'test_2.mov'

CONF_THRESHOLD = 0.4
INFERENCE_IMGSZ = int(os.getenv("YOLO_IMGSZ", "480"))
MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DET", "20"))

# ==========================================
# 2. 영어 클래스명 → 한국어 변환
# ==========================================
food_name_map = {

    'egg': '달걀',
    'kimchi': '김치',
    'lettuce': '상추',
    'pepper': '고추',
    'pork': '고기',
    'rice': '밥',
   
}

# ==========================================
# 음식 이름 정규화
# 사용자가 "계란"이라고 말해도 코드 내부 이름인 "달걀"로 바꿔줌
# ==========================================
def normalize_food_name(text):
    text = text.strip().replace(" ", "")

    alias_map = {
        "계란": "달걀",
        "달걀": "달걀",

        "김치": "김치",

        "상추": "상추",
        "쌈채소": "상추",

        "고추": "고추",

        "고기": "고기",
        "삼겹살": "고기",
        "돼지고기": "고기",
        "제육": "고기",

        "밥": "밥",
        "쌀밥": "밥",
        "공기밥": "밥",
    }

    for key, value in alias_map.items():
        if key in text:
            return value

    return text

# ==========================================
# 젓가락 끝점 기준으로 음식이 몇 시 방향인지 계산
# ==========================================
def get_clock_direction(chop_x, chop_y, food_x, food_y):
    dx = food_x - chop_x
    dy = food_y - chop_y

    # 너무 가까우면 방향 안내가 애매하므로 따로 처리
    if abs(dx) < 20 and abs(dy) < 20:
        return "바로 앞"

    # 화면 좌표계는 y가 아래로 갈수록 커짐
    # 그래서 위쪽을 12시로 만들기 위해 atan2(dx, -dy) 사용
    angle = math.degrees(math.atan2(dx, -dy))

    if angle < 0:
        angle += 360

    hour = round(angle / 30)

    if hour == 0:
        hour = 12

    return f"{hour}시"

# ==========================================
# OpenCV 한글 출력용 함수
# cv2.putText는 한글이 ????로 깨질 수 있어서 PIL로 출력
# ==========================================
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

try:
    korean_font = ImageFont.truetype(FONT_PATH, 22)
except:
    korean_font = None
    print("한글 폰트를 찾지 못했습니다. 영어 라벨로 출력합니다.")


def put_korean_text(frame, text, position, color=(0, 255, 0)):
    if korean_font is None:
        return frame

    # OpenCV는 BGR, PIL은 RGB라서 변환 필요
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # PIL은 RGB 색상 사용
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=korean_font, fill=rgb_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_korean_texts(frame, labels):
    if korean_font is None or not labels:
        return frame

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for text, position, color in labels:
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=korean_font, fill=rgb_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. 모델 및 영상 로드
# ==========================================
model = YOLO(model_path)

try:
    model.fuse()
except Exception as e:
    print("YOLO fuse skipped:", e)

detect_labels = set(food_name_map.keys()) | {"chopstick", "empty"}
model_names = (
    model.names.items()
    if hasattr(model.names, "items")
    else enumerate(model.names)
)
detect_classes = [
    cls_id
    for cls_id, name in model_names
    if name in detect_labels
]
if not detect_classes:
    detect_classes = None

video_path = "test_2.mov"

cap = cv2.VideoCapture(
    video_path
)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():

    print("❌ 영상을 열 수 없습니다.")

    exit()
# model = YOLO(model_path)

# # cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

# # 해상도 설정 (선택)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# if not cap.isOpened():
#     print("❌ 영상을 열 수 없습니다.")
#     exit()

print("🚀 영상 테스트를 시작합니다.")
print(" a : 음식 목록 출력")
print(" b : 현재 가리키는 음식 출력")
print(" c : 찾을 음식 음성 입력")
print(" q : 종료")

# ==========================================
# 4. 전역 변수
# ==========================================
# detected_foods = set()

current_target = ""
target_food = ""
last_vibration_time = 0

# ==========================================
# 5. 메인 루프
# ==========================================
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("✅ 영상 재생이 끝났습니다.")
        break

    # YOLO 추론
    results = model.predict(
        frame,
        imgsz=INFERENCE_IMGSZ,
        conf=CONF_THRESHOLD,
        classes=detect_classes,
        max_det=MAX_DETECTIONS,
        stream=True,
        verbose=False
    )

    for r in results:

        boxes = r.boxes

        best_chopstick = None

        max_conf = 0.0

        banchan_boxes = []

        current_foods = set()
        empty_detected = False
        korean_labels = []

        # ======================================
        # 6. 객체 처리
        # ======================================
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            # ----------------------------------
            # [A] 젓가락 처리
            # ----------------------------------
            if label == 'chopstick':

                if conf > CONF_THRESHOLD and conf > max_conf:

                    max_conf = conf

                    best_chopstick = {

                        'tip_x': int((x1 + x2) / 2),
                        'tip_y': y2,

                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }

            # ----------------------------------
            # [B] 음식 처리
            # ----------------------------------
            elif conf > CONF_THRESHOLD:

                # 1. 아예 처리하지 않을 객체들
                # hand, chopstick은 음식 아님
                # ssamjang은 요구사항상 출력/안내에서 제외
                if label in ['hand', 'chopstick', 'ssamjang']:
                    continue

                # 2. empty는 음식이 아니라 "빈 칸" 표시용
                # 실제 음식이 하나도 없을 때만 a 버튼에서 사용
                if label == 'empty':
                    empty_detected = True
                    continue

                # 3. food_name_map에 없는 객체는 음식으로 처리하지 않음
                if label not in food_name_map:
                    continue

                # 4. 영어 클래스명 → 한국어 음식명
                korean_name = food_name_map[label]

                # 5. 현재 프레임에서 보이는 음식 저장
                current_foods.add(korean_name)

                # 6. 음식 박스 정보 저장
                banchan_boxes.append({
                    'name': korean_name,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })

                # 7. 초록 박스 표시
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # 8. 한글 라벨 표시
                # 한글 폰트가 없으면 영어 label로 대체
                if korean_font is not None:
                    korean_labels.append((
                        f"{korean_name} {conf:.2f}",
                        (x1, y1 - 28),
                        (0, 255, 0)
                    ))
                else:
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        frame = put_korean_texts(frame, korean_labels)

        # ======================================
        # 7. 젓가락 충돌 판정
        # ======================================
        current_target = ""

        if best_chopstick:

            cx = best_chopstick['tip_x']

            cy = best_chopstick['tip_y']

            # 젓가락 박스
            cv2.rectangle(
                frame,
                (best_chopstick['x1'], best_chopstick['y1']),
                (best_chopstick['x2'], best_chopstick['y2']),
                (255, 0, 0),
                2
            )

            # 끝점 표시
            cv2.circle(
                frame,
                (cx, cy),
                8,
                (0, 0, 255),
                -1
            )

            # 충돌 검사
            for banchan in banchan_boxes:

               if (
                   banchan['x1'] < cx < banchan['x2']
                   and
                   banchan['y1'] < cy < banchan['y2']
                   ):
                   current_target = banchan['name']

                   if current_target == target_food:
                       
                       now = time.time()

                       if now - last_vibration_time > 1.5:
                           
                           GPIO.output(VIB_PIN, GPIO.HIGH)

                           time.sleep(0.3)

                           GPIO.output(VIB_PIN, GPIO.LOW)

                           last_vibration_time = now

    # ======================================
    # 8. 화면 출력
    # ======================================                                                                                                                                                                                               
    cv2.imshow("Capstone AIoT Project", frame)

    # ======================================
    # 9. 키 입력 처리
    # ======================================
    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------
    # [a] 음식 목록 출력
    # --------------------------------------
    if key == ord('a'):

        # 실제 음식이 하나라도 있으면 empty는 무시
        if len(current_foods) > 0:
            foods = ", ".join(sorted(current_foods))
            answer = f"현재 음식은 {foods} 입니다"

        # 실제 음식은 없고 empty만 인식된 경우
        elif empty_detected:
            answer = "현재 남아있는 음식이 없습니다"

        # empty도 음식도 제대로 인식 안 된 경우
        else:
            answer = "음식을 찾지 못했습니다"

        print(answer)
        speak_korean(answer)
    # --------------------------------------
    # [b] 현재 가리키는 음식 출력
    # --------------------------------------
    if key == ord('b'):

        if current_target == "":

            answer = "현재 가리키는 음식이 없습니다"

        else:

            answer = f"현재 {current_target} 을 가리키고 있습니다"

        print(answer)

        # os.system(
        #     f'espeak -v ko "{answer}" --stdout | aplay -D plughw:3,0'
        # )
        speak_korean(answer)

    # --------------------------------------
    # [c] 종료
    # --------------------------------------
    if key == ord('c'):
        answer = "찾고자 하는 음식의 이름을 말씀해주세요"
        print(answer)
        speak_korean(answer)

        try:
            text = listen_food()
            print("사용자 요청:", text)

            # 사용자가 말한 음식 이름 정리
            target_food = normalize_food_name(text)
            print("정규화된 음식 이름:", target_food)

            # 현재 화면에서 target_food에 해당하는 음식 찾기
            target_box = None

            for food in banchan_boxes:
                if food['name'] == target_food:
                    target_box = food
                    break

            # 음식이 현재 인식되지 않은 경우
            if target_box is None:
                answer = f"{target_food}은 현재 보이지 않습니다"
                print(answer)
                speak_korean(answer)

            # 젓가락이 인식되지 않은 경우
            elif best_chopstick is None:
                answer = "젓가락 위치를 찾지 못했습니다"
                print(answer)
                speak_korean(answer)

            # 음식도 있고 젓가락도 있는 경우 → 방향 계산
            else:
                chop_x = best_chopstick['tip_x']
                chop_y = best_chopstick['tip_y']

                food_x = int((target_box['x1'] + target_box['x2']) / 2)
                food_y = int((target_box['y1'] + target_box['y2']) / 2)

                direction = get_clock_direction(
                    chop_x,
                    chop_y,
                    food_x,
                    food_y
                )

                answer = f"{target_food}은 젓가락 기준 {direction} 방향에 있습니다"
                print(answer)
                speak_korean(answer)

        except Exception as e:
            print("음성 인식 오류:", e)
            speak_korean("음성을 인식하지 못했습니다")
    #==========================
    if key == ord('q'):
        print("🛑 프로그램 종료")
        break
    #===========================
    if key == ord('3'):
        os.system(
        "pactl set-sink-volume @DEFAULT_SINK@ +5%"
    )
        print("🔊 볼륨 증가")
    # 볼륨 감소 (노브 반시계방향 -> 1 입력)
    if key == ord('1'):
        os.system(
        "pactl set-sink-volume @DEFAULT_SINK@ -5%"
    )
        print("🔉 볼륨 감소")
# ==========================================
# 10. 종료 처리
# ==========================================
cap.release()

cv2.destroyAllWindows()

cap.release()

cv2.destroyAllWindows()

GPIO.cleanup()
#===========================================
# import cv2
# import os
# import asyncio
# import edge_tts
# import time
# import speech_recognition as sr
# import Jetson.GPIO as GPIO
# from ultralytics import YOLO

# # ==========================================
# # 음성 인식
# # ==========================================
# def listen_food():

#     recognizer = sr.Recognizer()

#     with sr.Microphone(device_index=0) as source:

#         print("🎤 마이크 준비")

#         recognizer.adjust_for_ambient_noise(
#             source,
#             duration=0.5
#         )

#         print("🎤 지금 말하세요")

#         audio = recognizer.listen(
#             source,
#             timeout=5,
#             phrase_time_limit=3
#         )

#         print("🎤 녹음 완료")

#     text = recognizer.recognize_google(
#         audio,
#         language="ko-KR"
#     )

#     return text


# # ==========================================
# # GPIO 설정
# # ==========================================
# VIB_PIN = 12

# GPIO.setmode(GPIO.BOARD)

# GPIO.setup(
#     VIB_PIN,
#     GPIO.OUT
# )


# # ==========================================
# # TTS
# # ==========================================
# async def speak(text):

#     communicate = edge_tts.Communicate(
#         text=text,
#         voice="ko-KR-SunHiNeural"
#     )

#     await communicate.save("voice.mp3")

#     # os.system(
#     #     "mpg123 -q voice.mp3"
#     # )
#     os.system(
#     "mpg123 -o alsa -a plughw:2,0 -q voice.mp3"
#     )


# def speak_korean(text):

#     try:

#         asyncio.run(
#             speak(text)
#         )

#     except Exception as e:

#         print(
#             "TTS 오류:",
#             e
#         )


# # ==========================================
# # YOLO 모델
# # ==========================================
# model_path = "best.pt"

# model = YOLO(
#     model_path
# )

# food_name_map = {

#     "egg": "달걀",
#     "kimchi": "김치",
#     "lettuce": "상추",
#     "pepper": "고추",
#     "pork": "고기",
#     "rice": "밥"

# }


# # ==========================================
# # 웹캠 설정 (USB 카메라)
# # ==========================================
# cap = cv2.VideoCapture(0)

# cap.set(
#     cv2.CAP_PROP_FRAME_WIDTH,
#     1280
# )

# cap.set(
#     cv2.CAP_PROP_FRAME_HEIGHT,
#     720
# )

# cap.set(
#     cv2.CAP_PROP_BUFFERSIZE,
#     1
# )

# if not cap.isOpened():

#     print("❌ 웹캠 열기 실패")

#     exit()


# print("🚀 실시간 음식 인식 시작")
# print("a : 음식 목록")
# print("b : 현재 가리키는 음식")
# print("c : 찾을 음식 음성 입력")
# print("q : 종료")


# # ==========================================
# # 변수
# # ==========================================
# detected_foods = set()

# current_target = ""

# target_food = ""

# last_vibration_time = 0

# frame_count = 0


# # ==========================================
# # 메인 루프
# # ==========================================
# while True:

#     success, frame = cap.read()

#     if not success:

#         print(
#             "카메라 읽기 실패"
#         )

#         break

#     frame_count += 1

#     # 프레임 스킵
#     if frame_count % 2 != 0:

#         continue

#     # ======================================
#     # YOLO 추론
#     # ======================================
#     results = model(

#         frame,

#         imgsz=640,

#         conf=0.4,

#         verbose=False

#     )

#     best_chopstick = None

#     max_conf = 0.0

#     banchan_boxes = []

#     # ======================================
#     # 객체 처리
#     # ======================================
#     for r in results:

#         for box in r.boxes:

#             x1, y1, x2, y2 = map(
#                 int,
#                 box.xyxy[0]
#             )

#             conf = float(
#                 box.conf[0]
#             )

#             cls_id = int(
#                 box.cls[0]
#             )

#             label = model.names[
#                 cls_id
#             ]

#             # ------------------------
#             # 젓가락
#             # ------------------------
#             if label == "chopstick":

#                 if conf > 0.4 and conf > max_conf:

#                     max_conf = conf

#                     best_chopstick = {

#                         "tip_x":
#                         int((x1+x2)/2),

#                         "tip_y":
#                         y2,

#                         "x1":
#                         x1,

#                         "y1":
#                         y1,

#                         "x2":
#                         x2,

#                         "y2":
#                         y2
#                     }

#             # ------------------------
#             # 음식
#             # ------------------------
#             elif label not in ["hand","chopstick"] and conf > 0.4:

#                 korean_name = food_name_map.get(

#                     label,

#                     label

#                 )

#                 detected_foods.add(

#                     korean_name

#                 )

#                 banchan_boxes.append({

#                     "name": korean_name,

#                     "x1": x1,
#                     "y1": y1,
#                     "x2": x2,
#                     "y2": y2

#                 })

#                 cv2.rectangle(

#                     frame,

#                     (x1,y1),

#                     (x2,y2),

#                     (0,255,0),

#                     2

#                 )

#                 cv2.putText(

#                     frame,

#                     f"{korean_name} {conf:.2f}",

#                     (x1,y1-10),

#                     cv2.FONT_HERSHEY_SIMPLEX,

#                     0.6,

#                     (0,255,0),

#                     2

#                 )

#     # ======================================
#     # 젓가락 충돌
#     # ======================================
#     current_target = ""

#     if best_chopstick:

#         cx = best_chopstick["tip_x"]

#         cy = best_chopstick["tip_y"]

#         cv2.circle(

#             frame,

#             (cx,cy),

#             8,

#             (0,0,255),

#             -1

#         )

#         for food in banchan_boxes:

#             if (

#                 food["x1"] < cx < food["x2"]

#                 and

#                 food["y1"] < cy < food["y2"]

#             ):

#                 current_target = food["name"]

#                 if current_target == target_food:

#                     now = time.time()

#                     if now - last_vibration_time > 1.5:

#                         GPIO.output(
#                             VIB_PIN,
#                             GPIO.HIGH
#                         )

#                         time.sleep(
#                             0.3
#                         )

#                         GPIO.output(
#                             VIB_PIN,
#                             GPIO.LOW
#                         )

#                         last_vibration_time = now

#     # ======================================
#     # 출력
#     # ======================================
#     cv2.imshow(

#         "Capstone AIoT Project",

#         frame

#     )

#     key = cv2.waitKey(1) & 0xFF

#     # 음식 목록
#     if key == ord("a"):

#         if len(detected_foods) == 0:

#             answer = "음식을 찾지 못했습니다"

#         else:

#             foods = ", ".join(

#                 detected_foods

#             )

#             answer = f"현재 음식은 {foods} 입니다"

#         print(answer)

#         speak_korean(answer)

#     # 현재 음식
#     if key == ord("b"):

#         if current_target == "":

#             answer = "현재 가리키는 음식이 없습니다"

#         else:

#             answer = f"{current_target} 을 가리키고 있습니다"

#         print(answer)

#         speak_korean(answer)

#     # 목표 음식 설정
#     if key == ord("c"):

#         speak_korean(
#             "찾고자 하는 음식 이름을 말씀해주세요"
#         )

#         try:

#             text = listen_food()

#             target_food = text.strip()

#             print(
#                 "사용자 요청:",
#                 target_food
#             )

#             speak_korean(

#                 f"{target_food} 위치를 안내합니다"

#             )

#         except:

#             speak_korean(

#                 "음성을 인식하지 못했습니다"

#             )

#     # 종료
#     if key == ord("q"):

#         break


# # ==========================================
# # 종료
# # ==========================================
# cap.release()

# cv2.destroyAllWindows()

# GPIO.cleanup()

# print(
#     "프로그램 종료"
# )
