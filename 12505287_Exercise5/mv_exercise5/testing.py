from pathlib import Path
import time
import re
from main import *

def compute_metrics(groundtruth, predicted_classes):
    gt_set = set(groundtruth)
    pred_set = set(predicted_classes.values())

    TP = len(gt_set & pred_set)
    FP = len(pred_set - gt_set)
    FN = len(gt_set - pred_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

def prepare_groundtruth(image):
    image_path = Path('groundtruth') / image

    groundtruth = []

    for f in image_path.iterdir():
        if f.is_file():
            name = f.stem
            name = re.sub(r'\d+$', '', name)
            groundtruth.append(name)

    return list(groundtruth)

def average_metrics(metrics_list):
    avg_metrics = {}

    for key in ['accuracy', 'precision', 'recall']:
        avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)

    return avg_metrics

def testing_on_all(whole_image = True, show_final_results=True):
    test_images = []

    results = []

    for i in range(10):
        test_images.append(f'image00{i}.pcd')

    for image in test_images:
        groundtruth = prepare_groundtruth(image)

        predicted_classes = pipeline(image, whole_image=whole_image, plots=False)

        print(groundtruth)
        print(predicted_classes.values())

        metrics = compute_metrics(groundtruth, predicted_classes)

        results.append(metrics)

    return results

if __name__ == '__main__':
    timings = {}

    t = time.perf_counter()
    # First methodd
    metrics = testing_on_all(whole_image=False, show_final_results=False)
    avg_metrics_1 = average_metrics(metrics)
    timings['first_method'] = time.perf_counter() - t

    t = time.perf_counter()
    # Second method
    metrics = testing_on_all(whole_image=False, show_final_results=False)
    avg_metrics_2 = average_metrics(metrics)
    timings['second_method'] = time.perf_counter() - t

    # Showing results
    print("Average metrics with the whole image:")
    for key, val in avg_metrics_1.items():
        print(f"{key} --> {val}")

    print("Average metrics with the segmented starting image:")
    for key, val in avg_metrics_2.items():
        print(f"{key} --> {val}")

    print("\n--- Execution times ---")
    for k, v in timings.items():
        print(f"{k:25s}: {v:.3f} s")
