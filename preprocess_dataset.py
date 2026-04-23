
# preprocess_dataset.py - 运行一次即可
import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# 导入 train.py 中的函数和常量
from train import (
    CocoKeypoints, make_heatmap, make_paf, 
    HEATMAP_SIZE, SIGMA, PAF_SIGMA, NUM_PAF_CHANNELS, NUM_JOINTS
)

def preprocess_and_save(data_dir, split="train2017", output_dir="./preprocessed"):
    """Preprocess dataset and save ground truth tensors."""
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = CocoKeypoints(data_dir, split)
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    print(f"Preprocessing {len(dataset)} images...")
    
    for idx in tqdm(range(len(dataset))):
        iid = dataset.image_ids[idx]

        # Get the aggregated tensors from dataset
        img, paf_t, hm_t, mask_t = dataset[idx]
        
        # Save as .npz file
        save_path = os.path.join(output_split_dir, f"{iid}.npz")
        np.savez(save_path, 
                 paf=paf_t.numpy(), 
                 hm=hm_t.numpy(), 
                 mask=mask_t.numpy())
    
    print(f"Preprocessed data saved to {output_split_dir}")

if __name__ == "__main__":
    preprocess_and_save("./data", "val2017")
    preprocess_and_save("./data", "train2017")