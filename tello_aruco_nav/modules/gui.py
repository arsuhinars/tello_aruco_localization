from collections import defaultdict
from enum import IntEnum
from time import time

import cv2
import numpy as np
import pygame as pg


class AlignHorizontal(IntEnum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1


class AlignVertical(IntEnum):
    TOP = -1
    CENTER = 0
    BOTTOM = 1


INITIAL_WINDOW_SIZE = (640, 480)
WINDOW_FRAMERATE = 30
CURSOR_SIZE = 40
TEXT_SPACING = 8
TEXT_PADDING = 12

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (255, 0, 0)


class Gui:
    def __init__(self):
        self.__is_running = False
        self.__image: np.ndarray | None = None
        self.__surface_image = np.zeros((*reversed(INITIAL_WINDOW_SIZE), 3), np.uint8)
        self.__image_time: float = 0.0
        self.__texts_map: dict[tuple[AlignHorizontal, AlignVertical], list[str]] = (
            defaultdict(list)
        )
        self.__pressed_keys: list[int] = []
        self.__clock = pg.time.Clock()

    @property
    def is_running(self):
        return self.__is_running

    def run(self):
        pg.display.init()
        self.__surface = pg.display.set_mode(INITIAL_WINDOW_SIZE, vsync=1)
        self.__is_running = True

    def stop(self):
        if not self.__is_running:
            return
        pg.display.quit()
        del self.__surface
        self.__is_running = False

    def push_image(self, img: np.ndarray | None):
        if img is not None:
            self.__image = img
            self.__image_time = time()

    def push_text(self, text: str, align_h: AlignHorizontal, align_v: AlignVertical):
        self.__texts_map[(align_h, align_v)].append(text)

    def update(self):
        if not self.__is_running:
            return

        self.__render()
        self.__handle_events()

        self.__clock.tick(WINDOW_FRAMERATE)

    def is_key_just_pressed(self, key: int):
        return key in self.__pressed_keys

    def __render(self):
        if time() - self.__image_time > 1.0:
            self.__image = None

        window_width, window_height = pg.display.get_window_size()
        if self.__image is not None:
            image_height, image_width, _ = np.shape(self.__image)
            if image_width != window_width or image_height != window_height:
                window_width = image_width
                window_height = image_height
                self.__surface = pg.display.set_mode((image_width, image_height))
                self.__surface_image = self.__image.copy()
            else:
                np.copyto(self.__surface_image, self.__image)

            cv2.line(
                self.__surface_image,
                ((image_width - CURSOR_SIZE) // 2, image_height // 2),
                ((image_width + CURSOR_SIZE) // 2, image_height // 2),
                WHITE_COLOR,
                2,
            )
            cv2.line(
                self.__surface_image,
                (image_width // 2, (image_height - CURSOR_SIZE) // 2),
                (image_width // 2, (image_height + CURSOR_SIZE) // 2),
                WHITE_COLOR,
                2,
            )
        else:
            self.__surface_image.fill(0)

            s = "Image is not available"
            (text_width, text_height), _ = cv2.getTextSize(
                s, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1
            )

            cv2.putText(
                self.__surface_image,
                s,
                (
                    (window_width - text_width) // 2,
                    (window_height - text_height) // 2,
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
                    self.__surface_image,
                    s,
                    (
                        int(
                            (window_width - text_width - TEXT_PADDING * 2.0) * h_factor
                            + TEXT_PADDING
                        ),
                        int(
                            (window_height - total_height - TEXT_PADDING * 2.0)
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

        pg.surfarray.blit_array(self.__surface, self.__surface_image.transpose(1, 0, 2))
        pg.display.flip()

    def __handle_events(self):
        self.__pressed_keys.clear()

        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    self.stop()
                case pg.KEYDOWN:
                    self.__pressed_keys.append(event.key)
