import os
import shutil
import tempfile
import argparse
import multiprocessing as mp
from tqdm import tqdm

import torch
from datasets import load_from_disk
from deepface import DeepFace
from PIL import Image


# =========================
# 配置
# =========================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

CONF_THRESHOLD = 0.6


# =========================
# DeepFace 情绪预测
# =========================

def predict_emotion(img_path, detector="retinaface"):
    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=detector
        )
        emotion_scores = result[0]["emotion"]
        label = max(emotion_scores, key=emotion_scores.get)
        conf = emotion_scores[label] / 100.0
        return label, conf
    except Exception: # 处理预测异常
        return None, 0.0 


def ensemble_emotion(img_path): # 集成多个检测器的预测结果
    preds = []

    for detector in ["retinaface", "mtcnn"]:
        label, conf = predict_emotion(img_path, detector)
        if label is not None:
            preds.append((label, conf))

    if len(preds) == 0:
        return None

    # 高置信度优先
    for label, conf in preds:
        if conf >= CONF_THRESHOLD:
            return label

    # 一致性判断
    labels = [p[0] for p in preds]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    return None


# =========================
# 处理一个 subset（train / validation）
# =========================

def process_subset(gpu_id, dataset_path, subset_name, output_root):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    dataset = load_from_disk(dataset_path)

    tmp_dir = tempfile.mkdtemp(prefix=f"fairface_tmp_gpu{gpu_id}_")

    for idx in tqdm(range(len(dataset)), desc=f"{subset_name} | GPU{gpu_id}"):
        sample = dataset[idx]
        img: Image.Image = sample["image"]

        tmp_img_path = os.path.join(tmp_dir, f"{subset_name}_{idx}.jpg")
        img.save(tmp_img_path)

        label = ensemble_emotion(tmp_img_path)
        if label is None:
            continue # 跳过无预测结果的样本

        save_dir = os.path.join(output_root, label)
        os.makedirs(save_dir, exist_ok=True)

        shutil.copy(tmp_img_path, os.path.join(save_dir, f"{subset_name}_{idx}.jpg"))

    shutil.rmtree(tmp_dir)


# =========================
# 主函数（双 GPU 并行）
# =========================

def main(args):
    os.makedirs(args.output, exist_ok=True)

    jobs = [
        (0, os.path.join(args.fairface_root, "train"), "train"),
        (1, os.path.join(args.fairface_root, "validation"), "validation"),
    ]

    processes = []
    for gpu_id, dataset_path, subset_name in jobs:
        p = mp.Process(
            target=process_subset,
            args=(gpu_id, dataset_path, subset_name, args.output)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fairface_root",
        type=str,
        required=True,
        help="Path to FairFace dataset root (contains train/ and validation/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Emotion_dataset",
        help="Output directory for emotion-labeled images"
    )

    args = parser.parse_args()
    main(args)

"""
python Face_Pose/test.py --fairface_root data/FairFace/fairface_0.25_parquet --output Face_Pose/Emotion_dataset

"""