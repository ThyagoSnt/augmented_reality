import cv2
import os
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, input_path, output_dir, ar_system):
        self.input_path = input_path
        self.output_dir = output_dir
        self.ar_system = ar_system

        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"processed_{base_name}")

        self.cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            output_path, fourcc, self.cap.get(cv2.CAP_PROP_FPS),
            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed = self.ar_system.process_frame(frame)
                self.out.write(processed)

                pbar.update(1)

        self.cap.release()
        self.out.release()
