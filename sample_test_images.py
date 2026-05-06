#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从四个数据集中采样头部/中部/尾部的图片到测试文件夹
"""

import os
import shutil
from pathlib import Path

def main():
    datasets = ["CLRS", "NWPU-RESISC45", "PatternNet", "RSSCN7"]
    
    # 创建测试目录结构
    test_root = Path("test_images")
    test_root.mkdir(exist_ok=True)
    
    for dataset in datasets:
        for category in ["head", "middle", "tail"]:
            (test_root / dataset / category).mkdir(parents=True, exist_ok=True)
    
    # 采样
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"处理 {dataset}")
        print(f"{'='*50}")
        
        data_path = Path("data") / dataset
        # 获取所有类别并排序
        classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        print(f"总类别数: {len(classes)}")
        
        # ===== 头部 =====
        print(f"\n📌 头部（前 3 个类别）:")
        for i in range(min(3, len(classes))):
            class_name = classes[i]
            class_path = data_path / class_name
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.tif"))
            sampled = images[:2]  # 取前 2 张
            print(f"  {class_name}: 复制 {len(sampled)} 张")
            for img in sampled:
                shutil.copy2(img, test_root / dataset / "head" / img.name)
        
        # ===== 中部 =====
        print(f"\n📌 中部（中间 2 个类别）:")
        mid_idx = len(classes) // 2 - 1
        for i in range(mid_idx, min(mid_idx + 2, len(classes))):
            class_name = classes[i]
            class_path = data_path / class_name
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.tif"))
            sampled = images[:1]  # 取前 1 张
            print(f"  {class_name}: 复制 {len(sampled)} 张")
            for img in sampled:
                shutil.copy2(img, test_root / dataset / "middle" / img.name)
        
        # ===== 尾部 =====
        print(f"\n📌 尾部（最后 3 个类别）:")
        for i in range(max(0, len(classes) - 3), len(classes)):
            class_name = classes[i]
            class_path = data_path / class_name
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.tif"))
            sampled = images[:2]  # 取前 2 张
            print(f"  {class_name}: 复制 {len(sampled)} 张")
            for img in sampled:
                shutil.copy2(img, test_root / dataset / "tail" / img.name)
    
    # 汇总统计
    print(f"\n{'='*50}")
    print("采样汇总统计:")
    print(f"{'='*50}")
    for dataset in datasets:
        head_count = len(list((test_root / dataset / "head").glob("*")))
        mid_count = len(list((test_root / dataset / "middle").glob("*")))
        tail_count = len(list((test_root / dataset / "tail").glob("*")))
        print(f"{dataset:20} | 头部: {head_count:2} | 中部: {mid_count:2} | 尾部: {tail_count:2}")
    
    print(f"\n✓ 采样完成！所有图片已保存到 {test_root}/")

if __name__ == "__main__":
    main()
