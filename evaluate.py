import os
from ultralytics import YOLO

if __name__ == '__main__':
    best_weights = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results', 'my_8n_run', 'weights', 'best.pt'
    )
    model = YOLO(best_weights)
    metrics = model.val(
        data='data.yaml',
        imgsz=640,
        batch=16,
        workers=0,
        save_json=False,
        plots=False,
        verbose=True
    )
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY — best.pt on validation set")
    print("="*70)
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("-"*70)
    print("Per-class breakdown:")
    print(f"  {'Class':<10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"  {'-'*44}")
    class_names = metrics.names
    for i, cls_name in class_names.items():
        p  = metrics.box.p[i]
        r  = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        ap   = metrics.box.ap[i]
        print(f"  {cls_name:<10} {p:>8.4f} {r:>8.4f} {ap50:>8.4f} {ap:>10.4f}")
    print("="*70)
