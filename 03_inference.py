#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/8/29 21:04
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_inference.py
# @Desc     :   

from pathlib import Path
from pydantic import validate_call, Field
from typing import Annotated, Literal, Union, Any

from ultralytics import YOLO


@validate_call
def load_pretrained_model(
        model_type: Annotated[
            Union[str, Literal["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]],
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
        item: Annotated[Union[str, Path], Field(description="The item to perform inference on")],
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
        name=f"conf_{confidence:.2f}",
        exist_ok=overwrite,
        stream=is_live,
        show=display,
    )
    if display:
        print(f"Results for {item}:")
        print_detective_results(results[0], model.names, display=display)


def main() -> None:
    """ Main Function """
    image: Path = Path("data/images/bus.png")
    print(image)

    model = load_pretrained_model("yolo26s.pt")

    inference_item(model, image, confidence=0.25, is_save=True, is_live=False, display=True)
    inference_item(model, image, confidence=0.75, is_save=True, is_live=False, display=True)

    help(inference_item)


if __name__ == "__main__":
    main()
