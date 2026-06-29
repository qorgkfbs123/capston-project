capstone_yolo
🥢 AIoT 기반 시각장애인 식사 보조 시스템 (Real-time Banchan Detector)
Python YOLOv8 OpenCV

📢 프로젝트 소개 (Project Overview)
본 프로젝트는 시각장애인이 식사 중 젓가락으로 가리킨 반찬을 실시간으로 인식하고 음성으로 안내하기 위해 개발된 컴퓨터 비전(Computer Vision) 시스템입니다.

Edge Computing 환경(NVIDIA Jetson Orin Nano)에서도 지연(Latency) 없이 매끄럽게 동작할 수 있도록 경량화 모델인 YOLOv8n을 채택하였으며, 자체적인 알고리즘을 통해 젓가락 끝점(Tip)과 객체 간의 충돌을 정밀하게 판정합니다.

✨ 주요 기능 (Key Features)
초고속 객체 인식: YOLOv8n 모델을 활용하여 밥상 위의 반찬(김치, 제육볶음, 계란 등)을 실시간으로 탐지합니다.
정밀 충돌 판정 알고리즘: 젓가락(Chopstick)의 바운딩 박스 하단 중심점을 계산하여, 해당 좌표가 반찬의 바운딩 박스 내부에 들어왔을 때만 이벤트를 발생시킵니다.
스마트 예외 처리 (Blacklist Filter): * 사용자의 손(hand)이나 안내가 불필요한 객체(ssamjang, empty 등)를 인식 결과에서 제외하여 오작동 및 불필요한 피드백을 방지합니다.
젓가락이 여러 개 인식될 경우, Confidence Score가 가장 높은 객체 1개만 추적하도록 커스텀 NMS 로직을 적용했습니다.
실시간 시연 녹화: cv2.VideoWriter를 통해 분석된 프레임 결과물을 원본과 동일한 FPS로 자동 저장합니다.
🛠 기술 스택 (Tech Stack)
Hardware: NVIDIA Jetson Orin Nano (Edge Device), NVIDIA RTX 4060 Ti (Model Training)
Software: Python 3.9.13, Ultralytics (YOLOv8), OpenCV
📂 파일 구조 (Repository Structure)
📦 Capstone-YOLO-Project
 ┣ 📜 best.pt           # Custom Dataset으로 학습된 YOLOv8n 최종 가중치 파일
 ┣ 📜 data.yaml         # 학습 데이터 경로 및 클래스(반찬 종류) 설정 파일
 ┣ 📜 main.py           # 실시간 웹캠/영상 테스트 및 충돌 판정 메인 실행 코드
 ┣ 📜 predict.py        # 단일 이미지 테스트용 코드
 ┣ 📜 train.py          # YOLOv8 모델 학습(Training) 실행 코드
 ┣ 📜 .gitignore        # Git 업로드 제외 파일 설정 (영상 및 데이터셋)
 ┗ 📜 README.md         # 프로젝트 설명서
