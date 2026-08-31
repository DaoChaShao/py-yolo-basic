#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/8/31 20:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_obb.py
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
                "yolo26n-cls", "yolo26s-cls", "yolo26m-cls", "yolo26l-cls", "yolo26x-cls",
                "yolo26n-obb", "yolo26s-obb", "yolo26m-obb", "yolo26l-obb", "yolo26x-obb",
                "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose",
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

    def __init__(
            self,
            model: YOLO,
            item: Any,
            *,
            accelerator: Union[str, Literal["cpu", "mps", "cuda"]] = "cpu",
            display: bool = True
    ) -> None:
        """
        Initialise the YOLO inference wrapper.

        :param model: An initialised YOLO model instance.
        :param item: Path to the image, video, or directory.
        :param accelerator: The accelerator to use for inference.
        :param display: Whether to display the inference results.
        :return: None
        """
        self._model = model
        self._item = item
        self._accelerator = accelerator
        self._display = display

    @validate_call
    def inference(
            self,
            *,
            task_type: Annotated[
                Union[str, Literal["detect", "obb", "prob"]],
                Field(default="detect", description="The task type")
            ] = "detect",
            confidence: Annotated[float, Field(ge=0.25, lt=1.0)] = 0.50,
            is_save: bool = True,
            overwrite: bool = True,
            is_live: bool = False,
    ) -> None:
        """
        Perform inference with a YOLO model on an image, a video, or a folder of images

        :param task_type: The task type.
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
            device=self._accelerator,
            show=self._display,
        )
        if self._display:
            print("*" * 65)
            print(f"Results for {self._item}:")
            print("-" * 65)
            match task_type:
                case "detect":
                    self._print_detection(_results, self._model.names)
                case "prob":
                    self._print_probability(_results)
                case "obb":
                    self._print_obb(_results)
                case _:
                    raise ValueError(f"Unsupported task type: {task_type}")
            print("*" * 65)

    @validate_call
    def _print_detection(
            self,
            results: Annotated[Any, Field(description="The results of the YOLO model")],
            names: Annotated[dict, Field(description="The names of the classes")],
    ) -> None:
        """
         Pretty-print detection results for a single item.

        :param results: The results of the YOLO model
        :param names: The names of the classes
        :return: None
        """
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print("No objects detected")
            return

        for i, box in enumerate(results[0].boxes):
            class_id: int = int(box.cls[0])
            conf: float = float(box.conf[0])
            coords: list = box.xyxy[0].tolist()
            model_name: str = names.get(class_id, f"{class_id}")
            print(f"Object {i + 1}: {model_name:12s} ({class_id}), Confidence: {conf:.4f}, Coordinates: {coords}")

    @validate_call
    def _print_probability(
            self,
            results: Annotated[Any, Field(description="The results of the YOLO model")],
    ) -> None:
        """
        Pretty-print probability results for a single item.

        :param results: The results of the YOLO model
        :return: None
        """
        probs = results[0].probs
        if probs is None:
            print("No probability results available. Ensure the model supports classification.")
            return

        top_class = int(probs.top1)
        top_confidence = float(probs.top1conf)
        class_name = self._model.names.get(top_class, str(top_class))
        print(f"Top class: {class_name} (ID: {top_class}), Confidence: {top_confidence:.4f}")

    @validate_call
    def _print_obb(
            self,
            results: Annotated[Any, Field(description="The results of the YOLO model")],
    ) -> None:
        """
        Pretty-print oriented bounding box (OBB) results for a single item.
        OBB results contain rotated rectangles with 4 corner points.

        :param results:
        :return: None
        """
        if results[0].obb is None or len(results[0].obb) == 0:
            print("No oriented objects detected")
            return

        _names = self._model.names
        print(f"Found {len(results[0].obb)} oriented object(s):\n")
        print(f"{'#':<4} {'Class':<20} {'Confidence':<12} {'Corners (x1,y1,x2,y2,x3,y3,x4,y4)'}")
        print("-" * 65)

        for i, obb in enumerate(results[0].obb):
            class_id = int(obb.cls[0])
            confidence = float(obb.conf[0])
            class_name = _names.get(class_id, str(class_id))

            corners = obb.xyxyxyxy[0].cpu().numpy().flatten()
            corners_str = ", ".join([f"({c:.1f})" for c in corners])
            print(f"{i + 1:<4} {class_name:<12} {confidence:.4f} {corners_str}")

    def __enter__(self) -> Self:
        """ Enter the runtime context for the context manager. """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """ Exit the runtime context. No cleanup needed. """
        pass

    def __repr__(self) -> str:
        """ Return a string representation of the Yolo object. """
        return f"<Yolo model={self._model} item={self._item} accelerator={self._accelerator}>"


def main() -> None:
    """ Main Function """
    quay: Path = Path("data/images/quay.png")

    model = load_pretrained_model("yolo26n-obb.pt", task_type="classify")
    with Yolo(model, quay, accelerator=detect_device(), display=True) as yolo:
        yolo.inference(task_type="obb")


if __name__ == "__main__":
    main()
