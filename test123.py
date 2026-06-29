# import speech_recognition as sr
# import os

# recognizer = sr.Recognizer()

# with sr.Microphone(device_index=0) as source:

#     print("🎤 주변 소음 측정 중...")
#     recognizer.adjust_for_ambient_noise(source, duration=1)

#     print("🎤 말하세요...")

#     audio = recognizer.listen(source)

#     try:
#         # 음성 → 텍스트 변환
#         text = recognizer.recognize_google(audio, language='ko-KR')

#         print("✅ 인식 결과:", text)

#         # 텍스트 → 스피커 출력
#         os.system(
#             f'espeak -v ko "{text}"'
#             )
#     except sr.UnknownValueError:
#         print("❌ 음성을 이해하지 못했습니다.")

#     except sr.RequestError:
#         print("❌ Google Speech 서비스 오류")
#===================================================
# import speech_recognition as sr

# for i, name in enumerate(sr.Microphone.list_microphone_names()):
#     print(i, name)
# import speech_recognition as sr

# r = sr.Recognizer()

# with sr.Microphone(device_index=0) as source:
#     print("🎤 말하세요")
#     audio = r.listen(source, timeout=5, phrase_time_limit=5)

# print("녹음 완료")

# from gtts import gTTS
# import asyncio
# import edge_tts
# import os

# async def speak(text):
#     communicate = edge_tts.Communicate(
#         text,
#         voice="ko-KR-SunHiNeural"
#     )

#     await communicate.save("voice.mp3")

# asyncio.run(
#     speak("현재 김치를 가리키고 있습니다")
# )

# os.system("mpg123 voice.mp3")
import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

VIB_PIN = 15   # IN 연결한 핀

GPIO.setup(
    VIB_PIN,
    GPIO.OUT
)

try:

    while True:

        GPIO.output(
            VIB_PIN,
            GPIO.HIGH
        )

        print("진동 ON")

        time.sleep(1)

        GPIO.output(
            VIB_PIN,
            GPIO.LOW
        )

        print("진동 OFF")

        time.sleep(1)

except KeyboardInterrupt:

    print("종료")

finally:

    GPIO.cleanup()
# import speech_recognition as sr

# r = sr.Recognizer()

# with sr.Microphone(device_index=0) as source:

#     print("말하세요")

#     audio = r.listen(
#         source,
#         timeout=10,
#         phrase_time_limit=5
#     )

# print("녹음 완료")

# text = r.recognize_google(
#     audio,
#     language="ko-KR"
# )

# print(text)