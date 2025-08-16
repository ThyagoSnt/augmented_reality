import cv2
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, input_path, output_path, ar_system):
        self.cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            output_path, fourcc, self.cap.get(cv2.CAP_PROP_FPS),
            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )
        self.ar_system = ar_system

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
