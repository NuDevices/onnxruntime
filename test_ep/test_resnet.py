import onnxruntime as ort
import numpy as np
from PIL import Image
import logging
import os
import time

MODEL_PATH = "models/resnet50_quantized.onnx"
IMAGE_PATH = f"images/{os.listdir('images')[10]}"
LABELS_PATH = "labels/synset_words.txt"
USE_NUDGEV = True
ENABLE_PROFILE = False
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_labels(labels_file):
    try:
        with open(labels_file, "r") as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            parts = line.strip().split(" ", 1)
            if len(parts) > 1:
                labels.append(parts[1])
            else:
                labels.append(parts[0])

        logger.debug(f"Loaded {len(labels)} labels from {labels_file}")
        return labels
    except Exception as e:
        logger.error(f"Error loading labels: {str(e)}")
        return ["Unknown"] * 1000


def preprocess_image_batch(image_path, input_shape, batch_size):
    target_height = input_shape[2]
    target_width = input_shape[3]

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
    num_labels = len(labels)
    logger.debug(f"Number of labels: {num_labels}")

    if not isinstance(outputs, list):
        outputs = [outputs]

    for output_idx, output in enumerate(outputs):
        logger.debug(f"Output {output_idx} shape: {output.shape}")

        batch_size = output.shape[0]
        for batch_idx in range(batch_size):
            scores = output[batch_idx]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()

            top_k = 5
            top_indices = np.argsort(probabilities)[-top_k:][::-1]

            print(f"\nPredictions for batch item {batch_idx+1}")
            print("-" * 50)
            print(f"{'Class':<40} {'Confidence':>10}")
            print("-" * 50)

            for idx in top_indices:
                idx_int = int(idx)
                if idx_int < num_labels:
                    label = labels[idx_int]
                    confidence = probabilities[idx_int]
                    print(f"{label:<40} {confidence:>10.4f}")
                else:
                    print(
                        f"[Index out of range: {idx_int}]{'':.<30} {probabilities[idx_int]:>10.4f}"
                    )


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
        if ENABLE_PROFILE:
            session_options.enable_profiling = True
        else:
            pass
        if USE_NUDGEV:
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
        output_name = session.get_outputs()[0].name

        logger.debug(f"Model input name: {input_name}, shape: {input_shape}")
        logger.debug(f"Model output name: {output_name}")

        labels = load_labels(LABELS_PATH)

        preprocess_start_time = time.time()
        batch_data = preprocess_image_batch(IMAGE_PATH, input_shape, batch_size)
        preprocess_time = time.time() - preprocess_start_time
        print(f"Preprocessing time: {preprocess_time:.3f} seconds")

        inference_times = []

        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}")

            inference_start_ns = time.perf_counter_ns()
            input_dict = {input_name: batch_data}
            outputs = session.run([output_name], input_dict)

            inference_end_ns = time.perf_counter_ns()
            inference_time_ns = inference_end_ns - inference_start_ns
            inference_times.append(inference_time_ns)

            inference_time_s = inference_time_ns / 1_000_000_000
            inference_time_ms = inference_time_ns / 1_000_000
            inference_time_us = inference_time_ns / 1_000

            print(f"Inference time: {inference_time_ns} ns")
            print(f"Inference time: {inference_time_us:.2f} μs")
            print(f"Inference time: {inference_time_ms:.3f} ms")
            print(f"Inference time: {inference_time_s:.6f} s")

            if run == 0 or run == num_runs - 1:
                show_predictions(outputs, labels)

        if num_runs > 1:
            avg_time = np.mean(inference_times) / 1_000_000
            std_time = np.std(inference_times) / 1_000_000
            min_time = np.min(inference_times) / 1_000_000
            max_time = np.max(inference_times) / 1_000_000

            print("\nInference time statistics (ms):")
            print(f"Average: {avg_time:.3f}")
            print(f"Std Dev: {std_time:.3f}")
            print(f"Min: {min_time:.3f}")
            print(f"Max: {max_time:.3f}")

        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.3f} seconds")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main(batch_size=8, num_runs=5)
