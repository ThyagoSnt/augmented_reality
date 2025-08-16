import os
import cv2
from tqdm import tqdm

class ImageBatchProcessor:
    def __init__(self, image_paths, output_dir, ar_system):
        self.image_paths = image_paths
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.ar_system = ar_system

    def run(self):
        for img_path in tqdm(self.image_paths, desc="Processing images"):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue

            processed = self.ar_system.process_frame(image)

            filename = os.path.basename(img_path)
            out_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(out_path, processed)
