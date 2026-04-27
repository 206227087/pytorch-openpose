"""Pre-compute ground truth heatmaps, PAFs, and masks for the COCO dataset.

Saves each image's ground truth as a .npz file for faster training.
Run once before training: python preprocess_dataset.py
"""

import os

import numpy as np
from tqdm import tqdm

from src.utils.CustomDataSet import CustomDataSet


def preprocess_and_save(data_dir, split="train2017", output_dir="./preprocessed"):
    """Preprocess dataset and save ground truth tensors."""
    os.makedirs(output_dir, exist_ok=True)

    dataset = CustomDataSet(data_dir, split)
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)

    print(f"Preprocessing {len(dataset)} images...")

    for idx in tqdm(range(len(dataset))):
        iid = dataset.image_ids[idx]
        img, paf_t, hm_t, mask_t = dataset[idx]

        # Get image path
        img_path = os.path.join(dataset.img_dir, dataset.id2file[iid])

        save_path = os.path.join(output_split_dir, f"{iid}.npz")

        np.savez_compressed(save_path,
                            img_path=np.array(img_path.encode('utf-8')),
                            paf=paf_t.numpy().astype(np.float16),
                            hm=hm_t.numpy().astype(np.float16),
                            mask=mask_t.numpy().astype(np.float16))

    print(f"Preprocessed data saved to {output_split_dir}")


if __name__ == "__main__":
    preprocess_and_save("./data", "val2017")
    preprocess_and_save("./data", "train2017")
