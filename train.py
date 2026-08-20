import os
from ultralytics import YOLO

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project=project_dir,
        name='my_8n_run',
        exist_ok=True,
        workers=0
    )


