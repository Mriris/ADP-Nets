import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='生成高斯噪声图像')
parser.add_argument('--input_dir', type=str, required=True, help='干净图像目录')
parser.add_argument('--output_dir', type=str, required=True, help='噪声图像输出目录')
parser.add_argument('--sigma', type=float, default=15, help='噪声标准差')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(('.jpg', '.png'))])
print(f'找到 {len(files)} 张图像，添加 sigma={args.sigma} 的高斯噪声')

for fname in tqdm(files):
    img = cv2.imread(os.path.join(args.input_dir, fname))
    img = img.astype(np.float32)

    # 添加高斯噪声
    noise = np.random.normal(0, args.sigma, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(args.output_dir, fname), noisy)

print(f'完成！噪声图像保存到 {args.output_dir}')
