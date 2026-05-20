"""Pre-compute ground truth heatmaps, PAFs, and masks for the COCO dataset.

Saves each image's ground truth as a .npz file for faster training.
Run once before training: python preprocess_dataset.py
"""

import os

import numpy as np
from tqdm import tqdm
from utils.HRNetCocoDataset import HRNetCocoDataset


def preprocess_and_save(data_dir, split="train2017", output_dir="../preprocessed"):
    """Preprocess dataset and save ground truth tensors."""
    os.makedirs(output_dir, exist_ok=True)

    dataset = HRNetCocoDataset(
        data_dir,
        split,
        input_size=256,
        heatmap_size=64,
        sigma=1.0,
        paf_sigma=2.0,
        augment=True
    )
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
                            paf=paf_t.numpy().astype(np.float32),
                            hm=hm_t.numpy().astype(np.float32),
                            mask=mask_t.numpy().astype(np.float32))

    print(f"Preprocessed data saved to {output_split_dir}")


if __name__ == "__main__":
    preprocess_and_save("../data", "train2017")
