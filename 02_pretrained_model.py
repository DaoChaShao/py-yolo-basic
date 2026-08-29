#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/8/29 20:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_pretrained_model.py
# @Desc     :   

from pathlib import Path
from pydantic import validate_call, Field
from typing import Literal, Annotated
from ultralytics import YOLO


@validate_call
def load_pretrained_model(
        model_type: Annotated[
            Literal["yolo26n.pt", "yolo26s.pt",],
            Field(default="yolo26n.pt", description="The type of YOLO model to load")
        ] = "yolo26n.pt",
        *,
        display: bool = False
) -> YOLO:
    """ Load a pretrained YOLO model """
    model: YOLO = YOLO(model_type)
    if display: print(model)
    return model


def main() -> None:
    """ Main Function """
    image: Path = Path("data/images/example.png")
    print(image)

    model = load_pretrained_model("yolo26s.pt")
    # print(model)

    result = model(image)
    result[0].show()


if __name__ == "__main__":
    main()
