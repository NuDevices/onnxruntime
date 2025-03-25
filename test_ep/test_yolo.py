import onnxruntime as ort
import numpy as np
from PIL import Image
import logging
import os
import time

MODEL_PATH = "models/yolov8s_quantized.onnx"
IMAGE_PATH = f"calibration_set/{os.listdir('calibration_set')[8]}"
LABELS_PATH = "labels/synset_words.txt"
nudgev = True
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        labels = [line.strip().split(" ", 1)[1] for line in f]
    logger.debug(f"Loaded {len(labels)} labels from {labels_file}")
    return labels


def preprocess_image_batch(image_path, input_shape, batch_size):
    target_height = 640
    target_width = 640

    logger.debug(f"Preprocessing image to size: {target_height}x{target_width}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_width, target_height), Image.BILINEAR)

    img_data = np.array(image).astype(np.float32)
    img_data = img_data / np.float32(255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean[None, None, :]) / std[None, None, :]

    img_data = img_data.transpose(2, 0, 1)
    batch_data = np.repeat(img_data[np.newaxis, :, :, :], batch_size, axis=0)

    logger.debug(
        f"Preprocessed batch shape: {batch_data.shape}, dtype: {batch_data.dtype}"
    )
    return batch_data


def show_predictions(outputs, labels):
    for i, output in enumerate(outputs):
        scores = output
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

        top_k = 5
        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        print(f"\nPredictions for batch item {i+1}")
        print("-" * 50)
        print(f"{'Class':<40} {'Confidence':>10}")
        print("-" * 50)

        for idx in top_indices:
            label = labels[idx]
            confidence = probabilities[idx]
            print(f"{label:<40} {confidence:>10.4f}")


def main(batch_size=1, num_runs=1):
    print("Available Execution Providers:", ort.get_available_providers())
    if "NudgevExecutionProvider" in ort.get_available_providers():
        print("Nudgev Execution Provider is correctly installed!")
    else:
        print("Warning: Nudgev EP not found among available providers")

    try:
        total_start_time = time.time()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.enable_profiling = False
        if nudgev:
            providers = ["NudgevExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        load_start_time = time.time()
        session = ort.InferenceSession(
            MODEL_PATH, sess_options=session_options, providers=providers
        )
        load_time = time.time() - load_start_time
        print(f"\nModel loading time: {load_time:.3f} seconds")

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        preprocess_start_time = time.time()

        batch_data = preprocess_image_batch(IMAGE_PATH, input_shape, batch_size)

        preprocess_time = time.time() - preprocess_start_time
        print(f"Preprocessing time: {preprocess_time:.3f} seconds")

        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}")

            inference_start_ns = time.perf_counter_ns()
            outputs = session.run(output_names, {input_name: batch_data})
            inference_end_ns = time.perf_counter_ns()
            print((outputs[0].flatten()[:30]))
            inference_time_ns = inference_end_ns - inference_start_ns

            inference_time_s = inference_time_ns / 1_000_000_000
            inference_time_ms = inference_time_ns / 1_000_000
            inference_time_us = inference_time_ns / 1_000

            print(f"Inference time: {inference_time_ns} ns")
            print(f"Inference time: {inference_time_us:.2f} μs")
            print(f"Inference time: {inference_time_ms:.3f} ms")
            print(f"Inference time: {inference_time_s:.6f} s")

        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.3f} seconds")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main(batch_size=8, num_runs=5)
