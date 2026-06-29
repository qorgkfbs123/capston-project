from ultralytics import YOLO

# 1. AI 뇌(모델) 불러오기: 가장 가볍고 빠른 YOLOv8 Small 모델을 뼈대로 사용합니다.
model = YOLO('yolov8s.pt') 

# 2. 윈도우 환경 에러 방지용 필수 코드
if __name__ == '__main__':
    
    # 3. 본격적인 학습 시작 명령!
    results = model.train(
        data='data.yaml',      # 데이터셋 정보가 담긴 파일 (같은 폴더에 있다고 가정)
        epochs=100,             # 전체 데이터셋을 몇 번 반복해서 공부할 것인가? (일단 100번)
        imgsz=640,             # 이미지를 640x640 크기로 리사이즈해서 학습 (YOLO 표준)
        batch=8,               # 그래픽카드에 한 번에 올릴 사진의 장수 (4060 Ti에 무리 없는 숫자)
        device=0,              # 0번 GPU (RTX 4060 Ti)를 사용하라는 명령
        name='capstone_v3'     # 학습 결과가 저장될 폴더 이름 지정
    )