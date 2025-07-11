import cv2
import sys
import serial
import time
import pygame
import serial.tools.list_ports
import os
import numpy as np
import mediapipe as mp
import random
import logging
import traceback
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 설정 상수 ---
SCALING_FACTOR = 4.0 # 점수 민감도 조절 (높을수록 작은 변화에도 점수가 크게 변함)

# --- 초기화 함수 ---

def init_face_detector(model_path):
    """Face Detector 모델을 초기화합니다."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    return vision.FaceDetector.create_from_options(options)

def init_camera():
    """카메라를 초기화하고 윈도우를 생성합니다."""
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    return cap

# --- 핵심 로직 함수 ---

def process_frame(image, face_detector):
    """단일 이미지 프레임을 처리하여 얼굴 감지 결과를 반환합니다."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return face_detector.detect(mp_image)

def calculate_score(detection, image_area):
    """감지된 얼굴 정보를 바탕으로 자세 점수를 계산합니다. 단순 점수 계산 함수, baseline, final 모두 이거로 계산"""
    bbox = detection.bounding_box
    
    # 1. Bbox 크기 점수 (얼굴이 가까울수록 점수 증가), 최대 100점
    bbox_area = bbox.width * bbox.height
    score_area = min((bbox_area / image_area) * 500, 100)

    # 2. 코 y좌표 점수 (코가 화면 아래로 갈수록 점수 증가), 
    score_nose = 0
    if len(detection.keypoints) > 2: # Keypoint 2번이 코
        score_nose = min(detection.keypoints[2].y * 120, 100) # y좌표가 클수록(아래로 갈수록) 점수 증가
    
    print(score_nose)


    # 최종 점수 (두 점수를 평균)
    total_score = (score_area + score_nose) / 2
    return total_score

def _update_and_get_scores(detection, image_area, baseline_score, baseline_set_time):
    """현재 점수를 계산하고, 기준 점수를 업데이트하며, 최종 점수를 반환합니다."""
    # 자세 점수는 매번 여기서 호출해서 계산
    raw_score = calculate_score(detection, image_area)

    # 만약 측정 안되어있으면 측정하고 time기록
    if baseline_score is None:
        baseline_score = raw_score
        baseline_set_time = time.time()
    
    # 최종 score은 현재 점수에서 base 점수 뺀 것
    final_score = raw_score - baseline_score
    
    # 점수를 0-100 범위로 정규화 및 클램핑
    final_score = max(0, final_score) # 음수 점수(기준보다 좋은 자세)는 0으로 처리
    final_score = final_score * SCALING_FACTOR # 점수 범위 확장
    final_score = min(100, final_score) # 100을 초과하지 않도록 클램핑

    return final_score, baseline_score, baseline_set_time

# --- UI 및 그리기 함수 ---

def draw_detection_info(image, detection, score):
    """감지된 얼굴의 경계 상자, 주요 지점, 점수를 그립니다."""
    # 경계 상자 그리기
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

    # 점수를 화면에 표시
    score_text = f"Posture Score: {score:.2f}"
    text_position = (start_point[0], start_point[1] - 10)
    cv2.putText(image, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 주요 지점(keypoints) 그리기
    for keypoint in detection.keypoints:
        keypoint_px = (int(keypoint.x * image.shape[1]), int(keypoint.y * image.shape[0]))
        cv2.circle(image, keypoint_px, 5, (255, 0, 0), -1)

def draw_ui_messages(image, baseline_score, baseline_set_time):
    """상황에 맞는 안내 메시지를 화면에 그립니다."""
    cv2.putText(image, "Press 'R' to reset Baseline Posture, make sure your keyboard setted in English", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if time.time() - baseline_set_time < 3:
        # 기준 자세 설정 직후 안내
        cv2.putText(image, "Baseline posture captured!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# --- 이벤트 핸들러 함수 ---

def _handle_input(key, baseline_score):
    """키 입력을 처리하고 프로그램 종료 여부 및 기준 점수 재설정 여부를 반환합니다."""
    should_exit = False
    reset_baseline = False

    if key == 27: # ESC 키
        should_exit = True
    elif key == ord('r') or key == ord('R'): # 'r' 또는 'R' 키
        reset_baseline = True
        print("Baseline reset by user.")
    
    return should_exit, reset_baseline

# --- 메인 실행 함수 ---

def main():
    """프로그램의 메인 실행 루프입니다."""
    face_detector = init_face_detector('blaze_face_short_range.tflite')
    cap = init_camera() # 카메라 초기화
    
    # 기즌 점수, 시간 초기화
    baseline_score = None
    baseline_set_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 프레임 얼굴 감지 결과 저장
        detection_result = process_frame(image, face_detector)

        if detection_result and detection_result.detections:
            image_area = image.shape[0] * image.shape[1]
            # 이미지 높이*너비, 추후 얼굴 박스 정규화에 사용

            # 첫 번째로 감지된 얼굴만 사용 (여러개 감지될 수 있음)
            detection = detection_result.detections[0]
            
            # 여기서 현재 최종 score, 기준자세 score, 기준 시간 모두 반환됨.
            final_score, baseline_score, baseline_set_time = _update_and_get_scores(
                detection, image_area, baseline_score, baseline_set_time
            )
            
            # 감지 정보 및 점수 그리기
            draw_detection_info(image, detection, final_score)
        
        # 안내 메시지 그리기
        draw_ui_messages(image, baseline_score, baseline_set_time)

        # 화면에 출력
        cv2.imshow('Face Detection', image)


        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        should_exit, reset_baseline = _handle_input(key, baseline_score)

        # R 눌리면 다시 기준자세 초기화, 다음 루프에서 다시 측정됨
        if reset_baseline:
            baseline_score = None

        # esc 눌리면 꺼짐
        if should_exit:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()