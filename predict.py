from ultralytics import YOLO

# 1. 방금 내가 직접 학습시킨 '최고의 뇌' 불러오기
model = YOLO('runs/detect/capstone_v2/weights/best.pt')

# 2. 무엇을 보고 테스트할 것인가? (아래 두 개 중 하나를 선택하세요!)
# [방법 A] 다운받았던 test 폴더 안의 사진 1장 경로를 적어줍니다. (이미지 이름은 본인 폴더에 있는 걸로 바꾸세요)
source_target = 'train/images/IMG_3067_MOV-0033_jpg.rf.3bd8aaa69f85a8f1c4e84cc24c76b312.jpg' 

# [방법 B] 내 컴퓨터(노트북)에 달린 웹캠으로 실시간으로 보기 (강력 추천!)
# source_target = 0  

# 3. AI야, 화면에 네모 박스를 그려줘!
results = model.predict(
    source=source_target, 
    show=True, 
    save=True,     
    name='v2_test_result',    # 폴더 이름도 헷갈리지 않게 v2용으로 고정!
    exist_ok=True             
)