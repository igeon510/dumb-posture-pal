
import cv2
import sys
import time
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 설정 상수 ---
SCALING_FACTOR = 4.0
ALERT_THRESHOLD = 50.0 # 자세 점수가 이 값을 넘으면 알림
ALERT_COOLDOWN = 5 # 알림을 보낸 후 최소 5초 대기

# --- 초기화 함수 ---
def init_face_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    return vision.FaceDetector.create_from_options(options)

# --- 핵심 로직 함수 ---
def process_frame(image, face_detector):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return face_detector.detect(mp_image)

def calculate_score(detection, image_area):
    bbox = detection.bounding_box
    bbox_area = bbox.width * bbox.height
    score_area = min((bbox_area / image_area) * 500, 100)
    score_nose = 0
    if len(detection.keypoints) > 2:
        score_nose = min(detection.keypoints[2].y * 120, 100)
    return (score_area + score_nose) / 2

def update_and_get_final_score(detection, image_area, baseline_score):
    raw_score = calculate_score(detection, image_area)
    if baseline_score is None:
        return 0, raw_score # 첫 프레임에서는 baseline을 설정하고 점수는 0으로 반환
    
    final_score = raw_score - baseline_score
    final_score = max(0, final_score)
    final_score = final_score * SCALING_FACTOR
    final_score = min(100, final_score)
    return final_score, baseline_score

def listen_for_commands(app_state):
    """표준 입력을 감지하여 명령을 처리하는 스레드 함수."""
    for line in sys.stdin:
        command = line.strip()
        if command == "RESET":
            app_state['reset_baseline'] = True

# --- 메인 실행 함수 ---
def main():
    try:
        # 공유 상태 객체
        app_state = {'reset_baseline': False}

        # 명령 리스너 스레드 시작
        command_thread = threading.Thread(target=listen_for_commands, args=(app_state,), daemon=True)
        command_thread.start()

        face_detector = init_face_detector('blaze_face_short_range.tflite')
        cap = cv2.VideoCapture(0)
        
        baseline_score = None
        last_alert_time = 0

        print("안녕하세요! Posture Pal이 당신의 자세를 감지하기 시작합니다. 바른 자세를 유지해주세요!", flush=True)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            detection_result = process_frame(image, face_detector)

            if detection_result and detection_result.detections:
                image_area = image.shape[0] * image.shape[1]
                detection = detection_result.detections[0]
                
                final_score, new_baseline = update_and_get_final_score(
                    detection, image_area, baseline_score
                )
                
                if baseline_score is None:
                    baseline_score = new_baseline
                    continue

                if app_state['reset_baseline']:
                    baseline_score = None
                    app_state['reset_baseline'] = False
                    print("Baseline reset by command.", flush=True)
                    continue

                # 임계값을 넘고 쿨다운이 지났으면 알림 전송
                if final_score > ALERT_THRESHOLD and (time.time() - last_alert_time) > ALERT_COOLDOWN:
                    print("POSTURE_ALERT", flush=True)
                    last_alert_time = time.time()
            
            # CPU 사용량을 줄이기 위해 잠시 대기
            time.sleep(0.1)

    except Exception as e:
        # 에러를 표준 출력으로 보내 Electron에서 확인할 수 있게 함
        print(f"ERROR: {e}", flush=True)
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

if __name__ == "__main__":
    main()
