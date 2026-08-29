#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/8/29 20:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_import.py
# @Desc     :   

import torch
import ultralytics


def check_yolo_status():
    """ Check YOLO and Torch status """
    print("*" * 65)
    print("YOLO and Torch status:")
    print("-" * 65)
    print(f"YOLO version: {ultralytics.__version__}")
    print("-" * 65)
    print(f"Torch version: {torch.__version__}")
    print(f"CPU available: {torch.cpu.is_available()}")
    print(f"Metal Performance Shaders available: {torch.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("*" * 65)


def main() -> None:
    """ Main Function """

    check_yolo_status()


if __name__ == "__main__":
    main()
