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
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("🎤 지금 말하세요")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
        print("🎤 녹음 완료")

    text = recognizer.recognize_google(audio, language="ko-KR")
    return text

#===========================================
import Jetson.GPIO as GPIO

VIB_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(VIB_PIN, GPIO.OUT)

#==========================================
from ultralytics import YOLO

# ==========================================
# Edge TTS
# ==========================================
async def speak(text):
    communicate = edge_tts.Communicate(text=text, voice="ko-KR-SunHiNeural")
    await communicate.save("voice.mp3")
    os.system("mpg123 -o alsa -a plughw:1,0 -q voice.mp3")

def speak_korean(text):
    try:
        asyncio.run(speak(text))
    except Exception as e:
        print("TTS 오류:", e)

# ==========================================
# 1. 기본 설정
# ==========================================
model_path = 'best.pt'

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

def normalize_food_name(text):
    text = text.strip().replace(" ", "")
    alias_map = {
        "계란": "달걀", "달걀": "달걀",
        "김치": "김치",
        "상추": "상추", "쌈채소": "상추",
        "고추": "고추",
        "고기": "고기", "삼겹살": "고기", "돼지고기": "고기", "제육": "고기",
        "밥": "밥", "쌀밥": "밥", "공기밥": "밥",
    }
    for key, value in alias_map.items():
        if key in text:
            return value
    return text

# ==========================================
# 젓가락 끝점 기준으로 음식이 몇 시 방향인지 계산 (사용자 기준)
# 맞은편 카메라 시점을 180도 뒤집어 사용자 시점으로 변환합니다.
# ==========================================
# ==========================================
# 젓가락 끝점 기준으로 음식이 몇 시 방향인지 계산
# ==========================================
def get_clock_direction(chop_x, chop_y, food_x, food_y):
    # 맨 처음 원본 코드 그대로 유지
    dx = food_x - chop_x
    dy = food_y - chop_y

    if abs(dx) < 20 and abs(dy) < 20:
        return "바로 앞"

    angle = math.degrees(math.atan2(dx, -dy))

    if angle < 0:
        angle += 360

    # 원본 시간 계산 (카메라 기준)
    hour = round(angle / 30)
    
    # 💡 사용자님의 아이디어 적용: 6시간을 더하고 12로 나눈 나머지를 구함
    # (예: 3시면 9시, 9시면 3시(15 % 12), 12시면 6시)
    hour = hour  % 12

    if hour == 0:
        hour = 12

    return f"{(hour+6)%12}시"


FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

try:
    korean_font = ImageFont.truetype(FONT_PATH, 22)
except:
    korean_font = None
    print("한글 폰트를 찾지 못했습니다. 영어 라벨로 출력합니다.")

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
# 3. 모델 및 웹캠 로드
# ==========================================
model = YOLO(model_path)

try:
    model.fuse()
except Exception as e:
    print("YOLO fuse skipped:", e)

detect_labels = set(food_name_map.keys()) | {"chopstick", "empty"}
model_names = model.names.items() if hasattr(model.names, "items") else enumerate(model.names)
detect_classes = [cls_id for cls_id, name in model_names if name in detect_labels]
if not detect_classes:
    detect_classes = None

# ✅ 동영상 파일 대신 웹캠(0번) 사용
cap = cv2.VideoCapture(0)

# ✅ 실시간 성능 확보를 위한 해상도 및 버퍼 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 지연(Lag) 방지를 위해 버퍼를 1로 최소화

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인해주세요.")
    exit()

print("🚀 실시간 웹캠 테스트를 시작합니다.")
print(" a : 음식 목록 출력")
print(" b : 현재 가리키는 음식 출력")
print(" c : 찾을 음식 음성 입력")
print(" q : 종료")

# ==========================================
# 4. 전역 변수
# ==========================================
current_target = ""
target_food = ""
last_vibration_time = 0

# ==========================================
# 5. 메인 루프
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("✅ 프레임을 읽어오지 못했습니다.")
        break

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

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == 'chopstick':
                if conf > CONF_THRESHOLD and conf > max_conf:
                    max_conf = conf
                    best_chopstick = {
                        'tip_x': int((x1 + x2) / 2),
                        'tip_y': y2,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    }

            elif conf > CONF_THRESHOLD:
                if label in ['hand', 'chopstick', 'ssamjang']:
                    continue

                if label == 'empty':
                    empty_detected = True
                    continue

                if label not in food_name_map:
                    continue

                korean_name = food_name_map[label]
                current_foods.add(korean_name)
                banchan_boxes.append({
                    'name': korean_name,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if korean_font is not None:
                    korean_labels.append((
                        f"{korean_name} {conf:.2f}",
                        (x1, y1 - 28),
                        (0, 255, 0)
                    ))
                else:
                    cv2.putText(
                        frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

        frame = put_korean_texts(frame, korean_labels)

        current_target = ""
        if best_chopstick:
            cx = best_chopstick['tip_x']
            cy = best_chopstick['tip_y']

            cv2.rectangle(frame, (best_chopstick['x1'], best_chopstick['y1']), 
                          (best_chopstick['x2'], best_chopstick['y2']), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            for banchan in banchan_boxes:
               if (banchan['x1'] < cx < banchan['x2'] and banchan['y1'] < cy < banchan['y2']):
                   current_target = banchan['name']
                   if current_target == target_food:
                       now = time.time()
                       if now - last_vibration_time > 1.5:
                           GPIO.output(VIB_PIN, GPIO.HIGH)
                           time.sleep(0.3)
                           GPIO.output(VIB_PIN, GPIO.LOW)
                           last_vibration_time = now

    cv2.imshow("Capstone AIoT Project", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        if len(current_foods) > 0:
            foods = ", ".join(sorted(current_foods))
            answer = f"현재 음식은 {foods} 입니다"
        elif empty_detected:
            answer = "현재 남아있는 음식이 없습니다"
        else:
            answer = "음식을 찾지 못했습니다"
        print(answer)
        speak_korean(answer)

    if key == ord('b'):
        if current_target == "":
            answer = "현재 가리키는 음식이 없습니다"
        else:
            answer = f"현재 {current_target} 을 가리키고 있습니다"
        print(answer)
        speak_korean(answer)

    if key == ord('c'):
        answer = "찾고자 하는 음식의 이름을 말씀해주세요"
        print(answer)
        speak_korean(answer)

        try:
            text = listen_food()
            print("사용자 요청:", text)

            target_food = normalize_food_name(text)
            print("정규화된 음식 이름:", target_food)

            target_box = None
            for food in banchan_boxes:
                if food['name'] == target_food:
                    target_box = food
                    break

            if target_box is None:
                answer = f"{target_food}은 현재 보이지 않습니다"
                print(answer)
                speak_korean(answer)

            elif best_chopstick is None:
                answer = "젓가락 위치를 찾지 못했습니다"
                print(answer)
                speak_korean(answer)

            else:
                chop_x = best_chopstick['tip_x']
                chop_y = best_chopstick['tip_y']
                food_x = int((target_box['x1'] + target_box['x2']) / 2)
                food_y = int((target_box['y1'] + target_box['y2']) / 2)

                direction = get_clock_direction(chop_x, chop_y, food_x, food_y)
                answer = f"{target_food}은 젓가락 기준 {direction} 방향에 있습니다"
                print(answer)
                speak_korean(answer)

        except Exception as e:
            print("음성 인식 오류:", e)
            speak_korean("음성을 인식하지 못했습니다")

    if key == ord('q'):
        print("🛑 프로그램 종료")
        break

    if key == ord('3'):
        os.system("pactl set-sink-volume @DEFAULT_SINK@ +5%")
        print("🔊 볼륨 증가")

    if key == ord('1'):
        os.system("pactl set-sink-volume @DEFAULT_SINK@ -5%")
        print("🔉 볼륨 감소")

# ==========================================
# 10. 종료 처리
# ==========================================
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()