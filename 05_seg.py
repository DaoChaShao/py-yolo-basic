#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/8/30 23:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   05_seg.py
# @Desc     :   

from pathlib import Path
from pydantic import validate_call, Field
from torch import cuda, backends
from typing import Annotated, Literal, Union, Any, Self

from ultralytics import YOLO


@validate_call
def load_pretrained_model(
        model_type: Annotated[
            Union[str, Literal[
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
                "YOLO26n-pose", "YOLO26s-pose", "YOLO26m-pose", "YOLO26l-pose", "YOLO26x-pose",
                "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt",
            ]],
            Field(default="yolo26n.pt", description="The type of YOLO model to load")
        ] = "yolo26n.pt",
        *,
        task_type: Annotated[
            Union[str, Literal["classify", "depth", "detect", "obb", "pose", "segment", "semantic"]],
            Field(default="detect", description="The task type")
        ] = "detect",
        display: bool = False
) -> YOLO:
    """
    Load a pretrained YOLO model

    :param model_type: The type of YOLO model to load
    :param task_type: The task type
    :param display: Whether to display the model information
    :return: The loaded YOLO model
    """
    model: YOLO = YOLO(model_type, task=task_type)
    if display: print(model)
    return model


@validate_call
def print_detective_results(
        results: Annotated[Any, Field(description="The results of the YOLO model")],
        names: Annotated[dict, Field(description="The names of the classes")],
        *,
        display: bool = True
) -> None:
    """
    Print detective results

    :param results: The results of the YOLO model
    :param names: The names of the classes
    :param display: Whether to display the results
    :return: None
    """
    if display:
        if results.boxes is None or len(results.boxes) == 0:
            print("No objects detected")
            return

        for i, box in enumerate(results.boxes):
            class_id: int = int(box.cls[0])
            conf: float = float(box.conf[0])
            coords: list = box.xyxy[0].tolist()
            model_name: str = names.get(class_id, f"{class_id}")
            print(f"Object {i + 1}: {model_name:16s} ({class_id}), Confidence: {conf:.4f}, Coordinates: {coords}")


@validate_call
def inference_item(
        model: Annotated[Any, Field(description="The YOLO model to use for inference")],
        item: Annotated[Any, Field(description="The item to perform inference on")],
        *,
        confidence: Annotated[float, Field(default=0.75, ge=0.25, lt=1.0)] = 0.50,
        is_save: bool = True,
        overwrite: bool = True,
        is_live: bool = False,
        display: bool = True
) -> None:
    """
    Perform inference with a YOLO model on an image, a video, or a folder of images

    :param model: The YOLO model to use for inference
    :param item: The image, video, screen, 0 (camera) or folder of images to perform inference on
    :param confidence: The confidence threshold for the detections
    :param is_save: Whether to save the inference results
    :param overwrite: Whether to overwrite the existing results
    :param is_live: Whether to perform inference in live mode
    :param display: Whether to display the inference results
    :return: None
    """
    results: Any = model.predict(
        source=item,
        conf=confidence,
        imgsz=640,
        save=is_save,
        name=f"conf-{confidence:.2f}",
        exist_ok=overwrite,
        stream=is_live,
        show=display,
    )
    if display:
        print(f"Results for {item}:")
        print_detective_results(results[0], model.names, display=display)


def detect_device(
        accelerator: str | Literal["auto", "cpu", "cuda", "mps"] = "auto",
        *,
        cuda_index: int = 0,
        display: bool = True
) -> str:
    """
    Detect and select the best available compute device.

    This function checks for CUDA, MPS (Apple Silicon), and falls back to CPU
    if no GPU is available. It provides detailed device information when display=True.

    :param accelerator: Target device type.
        - "auto": Automatically detect (CUDA > MPS > CPU)
        - "cuda": Use NVIDIA GPU (CUDA)
        - "mps": Use Apple Silicon GPU (MPS)
        - "cpu": Use CPU
    :param cuda_index: CUDA device index to use (default: 0).
    :param display: If True, prints detailed device information (default: True).
    :return: Device string compatible with PyTorch (e.g., "cuda:0", "mps", "cpu").
    """
    # Validate accelerator
    _accelerators: set[str] = {"auto", "cpu", "cuda", "mps"}
    if accelerator not in _accelerators:
        raise ValueError(f"Unsupported accelerator: {accelerator}. Must be one of {_accelerators}")

    # Handle CPU
    if accelerator == "cpu":
        if display:
            print("Using CPU as target device.")
        return "cpu"

    # Handle MPS
    if accelerator == "mps":
        if backends.mps.is_available():
            if display:
                print("Apple MPS device detected.")
            return "mps"
        if display:
            print("MPS unavailable. Falling back to CPU.")
        return "cpu"

    # Handle CUDA (for "auto" or "cuda")
    if cuda.is_available():
        count = cuda.device_count()

        # Validate CUDA index
        if cuda_index >= count:
            if display:
                print(f"CUDA device index {cuda_index} is out of range. Using 'cuda:0' instead.")
            cuda_index = 0

        # Display device info
        if display:
            print(f"Detected {count} CUDA GPU(s):")
            for i in range(count):
                print(f"GPU {i}: {cuda.get_device_name(i)}")
                print(f"- Allocated: {cuda.memory_allocated(i) / 1024 ** 3:.1f} GB")
                print(f"- Reserved:  {cuda.memory_reserved(i) / 1024 ** 3:.1f} GB")
            print(f"Using cuda:{cuda_index}")

        return f"cuda:{cuda_index}"

    # Fallback: No CUDA, check MPS for "auto"
    if accelerator == "auto" and backends.mps.is_available():
        if display:
            print("CUDA unavailable. Apple MPS device detected.")
        return "mps"

    # Final fallback: CPU
    if display:
        print("CUDA and MPS unavailable. Using CPU.")
    return "cpu"


class Yolo:
    """ A wrapper class for performing YOLO model """

    def __init__(self, model: YOLO, item: Any, *, display: bool = True) -> None:
        """
        Initialise the YOLO inference wrapper.

        :param model: An initialised YOLO model instance.
        :param item: Path to the image, video, or directory.
        :param display: Whether to display the inference results.
        :return: None
        """
        self._model = model
        self._item = item
        self._display = display

    @validate_call
    def inference(
            self,
            accelerator: Annotated[
                Union[str, Literal["cpu", "mps", "cuda"]],
                Field(default="cpu", description="The accelerator to use")
            ] = "cpu",
            *,
            confidence: Annotated[float, Field(default=0.75, ge=0.25, lt=1.0)] = 0.50,
            is_save: bool = True,
            overwrite: bool = True,
            is_live: bool = False,
    ) -> None:
        """
        Perform inference with a YOLO model on an image, a video, or a folder of images

        :param accelerator: The accelerator to use for inference.
        :param confidence: The confidence threshold for the detections.
        :param is_save: Whether to save the inference results.
        :param overwrite: Whether to overwrite the existing results.
        :param is_live: Whether to perform inference in live mode.
        :return: None
        """
        _results: Any = self._model.predict(
            source=self._item,
            conf=confidence,
            imgsz=640,
            save=is_save,
            name=f"conf-{confidence:.2f}",
            exist_ok=overwrite,
            stream=is_live,
            device=accelerator,
            show=self._display,
        )
        if self._display:
            print("*" * 65)
            print(f"Results for {self._item}:")
            print("-" * 65)
            self._print_dete_results(_results[0], self._model.names)
            print("*" * 65)

    @validate_call
    def _print_dete_results(
            self,
            results: Annotated[Any, Field(description="The results of the YOLO model")],
            names: Annotated[dict, Field(description="The names of the classes")],
    ) -> None:
        """
         Pretty-print detection results for a single item.

        :param results: The results of the YOLO model
        :param names: The names of the classes
        :param display: Whether to display the results
        :return: None
        """
        if self._display:
            if results.boxes is None or len(results.boxes) == 0:
                print("No objects detected")
                return

            for i, box in enumerate(results.boxes):
                class_id: int = int(box.cls[0])
                conf: float = float(box.conf[0])
                coords: list = box.xyxy[0].tolist()
                model_name: str = names.get(class_id, f"{class_id}")
                print(f"Object {i + 1}: {model_name:16s} ({class_id}), Confidence: {conf:.4f}, Coordinates: {coords}")

    def __enter__(self) -> Self:
        """ Enter the runtime context for the context manager. """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """ Exit the runtime context. No cleanup needed. """
        pass

    def __repr__(self) -> str:
        """ Return a string representation of the Yolo object. """
        return f"Yolo(model={self._model}, item={self._item})"


def main() -> None:
    """ Main Function """
    bus: Path = Path("data/images/bus.png")
    crossroads: Path = Path("data/images/crossroads.png")

    model = load_pretrained_model("yolo26n-seg.pt", task_type="segment")
    with Yolo(model, bus, display=True) as yolo:
        yolo.inference(detect_device())

    with Yolo(model, crossroads, display=True) as yolo:
        yolo.inference(detect_device())


if __name__ == "__main__":
    main()
