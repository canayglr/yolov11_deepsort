from ultralytics import YOLO
import time
import torch
import cv2
import numpy as np
from deep_sort.deep_sort import DeepSort
import random

model = YOLO("best.pt")

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

video_path = 'araba.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

unique_track_ids = set()
counter, fps_val = 0, 0
start_time = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    classes = ["araba"]
    results = model(rgb_frame, device=0, classes=[0], conf=0.8)

    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.xywh is None:
            continue

        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        tracks = tracker.update(xywh, conf, rgb_frame)

        for track in tracker.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()
            colors = [
            (0, 0, 255),     # Mavi
            (255, 0, 0),     # Kırmızı
            (0, 255, 0),     # Yeşil
            (255, 255, 0),   # Sarı
            (0, 255, 255),   # Cam göbeği
            (255, 0, 255),   # Mor
            (128, 0, 128),   # Morumsu
            (255, 165, 0),   # Turuncu
            (0, 128, 128),   # Teal
            (128, 128, 0)    # Zeytin yeşili
        ]

            color = colors[track_id % 10]
            cv2.rectangle(rgb_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(rgb_frame, f"araba-{track_id}", (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            unique_track_ids.add(track_id)

    current_time = time.perf_counter()
    elapsed = current_time - start_time
    counter += 1
    if elapsed >= 1.0:
        fps_val = counter / elapsed
        counter = 0
        start_time = current_time

    cv2.putText(rgb_frame, f"Araba Sayisi: {len(unique_track_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    cv2.imshow("Video", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
