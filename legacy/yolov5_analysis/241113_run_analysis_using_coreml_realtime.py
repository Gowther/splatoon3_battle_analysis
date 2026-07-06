#%%
import warnings
# Deprecation warningを抑制
warnings.filterwarnings('ignore', category=FutureWarning)

import datetime
from collections import deque
import csv
import random
import glob
import numpy as np
import cv2
import statistics
import os
from PIL import Image
import time
import re
import coremltools as ct
import torch

#%%
# Core MLモデルのロード
def load_coreml_models():
    model = ct.models.MLModel('../models/the_model.mlpackage/Data/com.apple.CoreML/model.mlmodel')
    ocr_model = ct.models.MLModel('../models/ocr_model.mlpackage/Data/com.apple.CoreML/model.mlmodel')
    message_ocr_model = ct.models.MLModel('../models/message_ocr_model.mlpackage/Data/com.apple.CoreML/model.mlmodel')
    return model, ocr_model, message_ocr_model

# Core ML用の推論関数
def inference_frame(model, frame):
    # 记录原始帧尺寸
    original_height, original_width = frame.shape[:2]
    
    # 入力画像の前処理
    input_image = cv2.resize(frame, (640, 640))
    # OpenCV(BGR)からPIL(RGB)に変換
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)
    
    # 推論実行
    input_dict = {'input_image': input_image}
    results = model.predict(input_dict)
    
    return standardize_results(results, original_width, original_height)

# Core MLの出力を標準化する関数
def standardize_results(coreml_output, original_width, original_height):
    # Core MLの出力から検出結果を取得
    output_array = coreml_output['var_833'][0]
    
    # 计算缩放比例
    scale_x = original_width / 640
    scale_y = original_height / 640
    
    # 検出結果を変換
    detections = []
    for detection in output_array:
        # 最初の4つの値がバウンディングボックスの座標
        x1, y1, x2, y2 = detection[:4]
        # 将坐标映射回原始尺寸
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y
        # 残りの値から最大の確信度とそのクラスインデックスを取得
        confidence_values = detection[4:]
        class_id = np.argmax(confidence_values)
        confidence = confidence_values[class_id]
        
        if confidence > 0.98:  # 確信度のしきい値
            detections.append([x1, y1, x2, y2, confidence, class_id])
    
    # numpy配列に変換
    detections = np.array(detections) if detections else np.zeros((0, 6))
    
    # PyTorch YOLOv5の出力形式に合わせたオブジェクトを作成
    class Results:
        def __init__(self, detections, original_img):
            self.xyxy = [torch.from_numpy(detections)]
            self.ims = [original_img]
            self._names = None
        
        @property
        def names(self):
            if self._names is None:
                self._names = {
                    0: 'alive',
                    1: 'area_object',
                    2: 'asari_object',
                    3: 'dead',
                    4: 'enemy_gears',
                    5: 'fixed_count',
                    6: 'hoko_canmon',
                    7: 'ika_player',
                    8: 'kill_log',
                    9: 'lead',
                    10: 'map_enemy_info',
                    11: 'map_info',
                    12: 'map_player_dead',
                    13: 'map_player_position',
                    14: 'message',
                    15: 'moving_count',
                    16: 'object',
                    17: 'other_player',
                    18: 'penalty',
                    19: 'player',
                    20: 'special',
                    21: 'special_state',
                    22: 'timer',
                    23: 'yagura_kanmon',
                    24: 'unknow'
                }
            return self._names
        
        def print(self):
            for det in self.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det
                print(f"Class: {self.names[int(cls)]}, Confidence: {conf:.2f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    return Results(detections, coreml_output.get('input_image'))

#%%
# メインのリアルタイム処理ループ
def main():
    # モデルのロード
    model, ocr_model, message_ocr_model = load_coreml_models()
    
    # 動画ファイルのリスト取得
    video_files = glob.glob('../footages/*.mp4')
    print(f"Found {len(video_files)} video files")
    random.shuffle(video_files)

    for input_video_path in video_files:
        print(f"Processing: {input_video_path}")
        csv_path = input_video_path.split(".")[0] + "_coreml.csv"
        
        if os.path.isfile(csv_path):
            print(f"Skipping: {csv_path} already exists")
            continue

        # ビデオキャプチャの設定
        cap = cv2.VideoCapture(input_video_path)
        fps = 5
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // fps)
        frame_count = 0
        final_result = deque()

        while True:
            count = 0
            while count < frame_interval:
                ret = cap.grab()
                if not ret:
                    break
                count += 1

            if not ret:
                break

            # フレーム読み込みと処理時間計測
            total_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 推論実行
            results = inference_frame(model, frame)
            print("Core ML output format:")
            results.print()
            
            # 将检测结果添加到 final_result
            detections = results.xyxy[0].cpu().numpy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                result_row = [frame_count, int(cls), conf, x1, y1, x2, y2]
                final_result.append(result_row)
            
            # プレビュー表示（デバッグ用）
            preview_frame = frame.copy()
            
            # 在预览帧上绘制检测结果
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # 转换坐标为整数
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # 绘制矩形框
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 添加类别名称和置信度
                label = f"{results.names[int(cls)]} {conf:.2f}"
                # 计算文本大小
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                # 绘制文本背景
                cv2.rectangle(preview_frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 255, 0), -1)
                # 绘制文本
                cv2.putText(preview_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            preview_scale = 0.5
            preview_frame_resized = cv2.resize(preview_frame, None, 
                                             fx=preview_scale, 
                                             fy=preview_scale)
            cv2.imshow("Core ML Detection", preview_frame_resized)

            # 処理時間の計算と表示
            total_time = time.time() - total_start
            print(f"\nFrame {frame_count}:")
            print(f"Total processing time: {total_time*1000:.1f}ms")
            print(f"FPS: {1/total_time:.1f}")

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # 結果の保存
        final_result = list(final_result)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for res in final_result:
                writer.writerow(res)

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()