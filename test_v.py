import Jetson.GPIO as GPIO
import time

# 신호선(IN)을 변경된 11번 핀으로 설정 (BOARD 번호 기준)
IN_PIN = 11 

def main():
    # 핀 번호 체계를 BOARD 번호(물리적 핀 순서)로 설정
    GPIO.setmode(GPIO.BOARD)
    
    # 11번 핀을 출력(OUT) 모드로 설정하고 초기 상태는 꺼짐(LOW)으로 지정
    GPIO.setup(IN_PIN, GPIO.OUT, initial=GPIO.LOW)

    print("11번 핀으로 진동 모터 테스트 시작... (종료하려면 CTRL+C)")
    
    try:
        while True:
            print("진동 ON!")
            GPIO.output(IN_PIN, GPIO.HIGH)  # 11번 핀에 3.3V 신호 출력 (모터 작동)
            time.sleep(1)                   # 1초 유지
            
            print("진동 OFF!")
            GPIO.output(IN_PIN, GPIO.LOW)   # 11번 핀에 0V 신호 출력 (모터 정지)
            time.sleep(1)                   # 1초 유지
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중단되었습니다.")
        
    finally:
        # 프로그램이 끝나면 사용한 GPIO 핀을 안전하게 초기화
        GPIO.cleanup()
        print("GPIO 설정이 초기화되었습니다.")

if __name__ == '__main__':
    main()