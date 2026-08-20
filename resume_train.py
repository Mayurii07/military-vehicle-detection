import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Resume training from the last checkpoint
    last_weights = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results', 'my_8n_run', 'weights', 'last.pt'
    )
    model = YOLO(last_weights)
    model.train(resume=True)
