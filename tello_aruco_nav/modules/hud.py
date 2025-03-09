from collections import defaultdict
from enum import IntEnum
from time import time

import cv2
import numpy as np


class AlignHorizontal(IntEnum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1


class AlignVertical(IntEnum):
    TOP = -1
    CENTER = 0
    BOTTOM = 1


HUD_FRAMERATE = 30
INITIAL_IMAGE_SIZE = (640, 480)
CURSOR_SIZE = 40
TEXT_SPACING = 8
TEXT_PADDING = 12

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (255, 0, 0)


class Hud:
    def __init__(self):
        self.__input_image: np.ndarray | None = None
        self.__buffer_image = np.zeros((*reversed(INITIAL_IMAGE_SIZE), 3), np.uint8)
        self.__result_image = np.zeros((*reversed(INITIAL_IMAGE_SIZE), 3), np.uint8)
        self.__image_time: float = 0.0
        self.__texts_map: dict[tuple[AlignHorizontal, AlignVertical], list[str]] = (
            defaultdict(list)
        )

    @property
    def image(self):
        return self.__result_image

    def push_image(self, img: np.ndarray | None):
        if img is not None:
            self.__input_image = img
            self.__image_time = time()

    def push_text(self, text: str, align_h: AlignHorizontal, align_v: AlignVertical):
        self.__texts_map[(align_h, align_v)].append(text)

    def update(self):
        if time() - self.__image_time > 1.0:
            self.__input_image = None

        target_height, target_width, _ = np.shape(self.__buffer_image)
        if self.__input_image is not None:
            input_height, input_width, _ = np.shape(self.__input_image)
            if input_width != target_width or input_height != target_height:
                target_width = input_width
                target_height = input_height
                self.__buffer_image = self.__input_image.copy()
            else:
                np.copyto(self.__buffer_image, self.__input_image)

            cv2.line(
                self.__buffer_image,
                ((input_width - CURSOR_SIZE) // 2, input_height // 2),
                ((input_width + CURSOR_SIZE) // 2, input_height // 2),
                WHITE_COLOR,
                2,
            )
            cv2.line(
                self.__buffer_image,
                (input_width // 2, (input_height - CURSOR_SIZE) // 2),
                (input_width // 2, (input_height + CURSOR_SIZE) // 2),
                WHITE_COLOR,
                2,
            )
        else:
            self.__buffer_image.fill(0)

            s = "Image is not available"
            (text_width, text_height), _ = cv2.getTextSize(
                s, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1
            )

            cv2.putText(
                self.__buffer_image,
                s,
                (
                    (target_width - text_width) // 2,
                    (target_height - text_height) // 2,
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                RED_COLOR,
                1,
            )

        for (align_h, align_v), texts in self.__texts_map.items():
            total_height = 0.0
            for s in texts:
                (_, text_height), _ = cv2.getTextSize(
                    s, cv2.FONT_HERSHEY_PLAIN, 1.25, 1
                )
                total_height += text_height + TEXT_SPACING
            total_height -= TEXT_SPACING

            h_factor = (align_h + 1.0) / 2.0
            v_factor = (align_v + 1.0) / 2.0
            offset_y = 0
            for s in texts:
                (text_width, text_height), _ = cv2.getTextSize(
                    s, cv2.FONT_HERSHEY_PLAIN, 1.25, 1
                )
                cv2.putText(
                    self.__buffer_image,
                    s,
                    (
                        int(
                            (target_width - text_width - TEXT_PADDING * 2.0) * h_factor
                            + TEXT_PADDING
                        ),
                        int(
                            (target_height - total_height - TEXT_PADDING * 2.0)
                            * v_factor
                            + TEXT_PADDING
                        )
                        + offset_y
                        + text_height,
                    ),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.25,
                    WHITE_COLOR,
                    1,
                )

                offset_y += text_height + TEXT_SPACING
            texts.clear()

        if self.__result_image.shape != self.__buffer_image.shape:
            self.__result_image = self.__buffer_image.copy()
        else:
            np.copyto(self.__result_image, self.__buffer_image)
