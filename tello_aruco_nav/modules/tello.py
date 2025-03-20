import asyncio
import logging
import re
import socket
from collections import defaultdict
from enum import IntEnum
from threading import Thread
from time import time

import av
import numpy as np
from av.codec.hwaccel import HWAccel
from av.video.reformatter import VideoReformatter

from tello_aruco_nav.common.exceptions import (
    TelloAlreadyConnectedException,
    TelloDisconnectedException,
    TelloErrorException,
    TelloFailedConnectException,
)

logger = logging.getLogger("tello")

TELLO_IP = "192.168.10.1"
TELLO_COMMAND_PORT = 8889
TELLO_STATE_PORT = 8890
TELLO_STREAM_PORT = 11111
TELLO_ADDRESS = (TELLO_IP, TELLO_COMMAND_PORT)
TIMEOUT_DELAY = 5.0
MAX_RETRIES_COUNT = 3
TELLO_STATE_REGEX = re.compile(r"(\w+):(\d*(?:.\d{1,2})?)(?:;|$)")
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480


class TelloConnectionState(IntEnum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2


class Tello:
    def __init__(self):
        self.__connection_state = TelloConnectionState.DISCONNECTED
        self.__is_streaming = False
        self.__is_flying = False
        self.__lock = asyncio.Lock()
        self.__response_event = asyncio.Event()
        self.__last_frame: np.ndarray | None = None
        self.__last_frame_time = 0.0
        self.__state_dict: dict[str, float] = defaultdict(lambda: 0.0)

    @property
    def connection_state(self):
        return self.__connection_state

    @property
    def is_streaming(self):
        return self.__is_streaming

    @property
    def is_flying(self):
        return self.__is_flying

    @property
    def battery(self):
        return int(self.__state_dict["bat"])

    @property
    def height(self):
        return self.__state_dict["tof"] / 100.0

    async def connect(self):
        if self.__connection_state != TelloConnectionState.DISCONNECTED:
            raise TelloAlreadyConnectedException()

        logger.info("Connecting")

        self.__cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.__cmd_sock.bind(("", TELLO_COMMAND_PORT))

        self.__state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.__state_sock.bind(("", TELLO_STATE_PORT))
        self.__state_sock.settimeout(TIMEOUT_DELAY)

        self.__connection_state = TelloConnectionState.CONNECTING
        self.__is_streaming = False
        self.__is_flying = False
        self.__last_frame = None
        self.__response_event.clear()

        try:
            self.__cmd_sock.sendto(b"command", TELLO_ADDRESS)
        except Exception:
            self.__on_disconnected()
            raise TelloFailedConnectException()

        async with self.__lock:
            self.__response_thread = Thread(
                target=self.__run_response_thread, daemon=True
            )
            self.__status_thread = Thread(target=self.__run_state_thread, daemon=True)
            self.__response_thread.start()
            self.__status_thread.start()

            try:
                async with asyncio.timeout(TIMEOUT_DELAY):
                    await self.__response_event.wait()
            except asyncio.TimeoutError:
                self.__on_disconnected()
                logger.error("Failed to connect")
                raise TelloFailedConnectException()

        self.__connection_state = TelloConnectionState.CONNECTED

        logger.info("Connected")

    def disconnect(self):
        self.__on_disconnected()
        logger.info("Disconnected")

    async def stream_on(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        await self.__send_command(b"streamon")
        self.__av_container = av.open(
            f"udp://0.0.0.0:{TELLO_STREAM_PORT}",
            format="h264",
            timeout=TIMEOUT_DELAY,
            buffer_size=4096,
            hwaccel=HWAccel(device_type="vaapi", device="/dev/dri/renderD128"),
        )
        self.__is_streaming = True
        self.__av_thread = Thread(target=self.__run_stream_thread, daemon=True)
        self.__av_thread.start()

        logger.info("Stream on")

    def stream_off(self):
        if (
            self.__connection_state != TelloConnectionState.CONNECTED
            or not self.__is_streaming
        ):
            return

        self.__send_command_no_response(b"streamoff")

        del self.__av_thread

        self.__is_streaming = False

        logger.info("Stream off")

    async def takeoff(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        await self.__send_command(b"takeoff")
        await asyncio.sleep(5.0)
        self.__is_flying = True

    async def land(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        await self.__send_command(b"land")
        await asyncio.sleep(3.0)
        self.__is_flying = False

    def emergency(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__is_flying = False
        self.__send_command_no_response(b"emergency")

    def send_rc_control(
        self, left_right: int, forward_backward: int, up_down: int, yaw: int
    ):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command_no_response(
            f"rc {left_right} {forward_backward} {up_down} {yaw}".encode("ascii")
        )

    async def go(self, x: int, y: int, z: int, speed: int):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        await self.__send_command(f"go {x} {y} {z} {speed}".encode("ascii"))
        await asyncio.sleep(3.0)

    def get_video_frame(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            return None

        if time() - self.__last_frame_time > TIMEOUT_DELAY:
            return None

        return self.__last_frame

    async def __send_command(self, command: bytes):
        async with self.__lock:
            self.__cmd_sock.sendto(command, TELLO_ADDRESS)
            i = 0
            while i < MAX_RETRIES_COUNT:
                try:
                    async with asyncio.timeout(TIMEOUT_DELAY):
                        await self.__response_event.wait()
                        break
                except asyncio.TimeoutError:
                    if self.__connection_state == TelloConnectionState.DISCONNECTED:
                        raise TelloDisconnectedException()
                    i += 1
                    if i == MAX_RETRIES_COUNT:
                        self.__on_disconnected()
                        raise TelloDisconnectedException()

    def __send_command_no_response(self, command: bytes):
        self.__cmd_sock.sendto(command, TELLO_ADDRESS)

    def __on_disconnected(self):
        if self.__connection_state == TelloConnectionState.DISCONNECTED:
            return

        self.__connection_state = TelloConnectionState.DISCONNECTED
        self.__is_flying = False

        try:
            if self.__is_streaming:
                self.__cmd_sock.sendto(b"streamoff", TELLO_ADDRESS)
            self.__cmd_sock.sendto(b"emergency", TELLO_ADDRESS)
        except Exception:
            logger.exception("Failed to send emergency message during disconnecting")

        self.__cmd_sock.close()
        self.__state_sock.close()

        del self.__cmd_sock
        del self.__state_sock

        if self.__is_streaming:
            self.__is_streaming = False
            del self.__av_thread

    def __run_response_thread(self):
        try:
            while self.__connection_state != TelloConnectionState.DISCONNECTED:
                match self.__cmd_sock.recv(8):
                    case b"ok":
                        self.__response_event.set()
                    case b"error":
                        self.__on_disconnected()
                        raise TelloErrorException()
        except Exception:
            self.__on_disconnected()
            logger.exception("Response thread stopped")

    def __run_state_thread(self):
        try:
            while self.__connection_state != TelloConnectionState.DISCONNECTED:
                try:
                    state_msg = self.__state_sock.recv(1024).decode("ascii")
                    for m in TELLO_STATE_REGEX.finditer(state_msg):
                        self.__state_dict[m.group(1)] = float(m.group(2))
                except TimeoutError:
                    self.__on_disconnected()
                    logger.error("State timeout exceeded. Disconnected")
                    break
        except Exception:
            self.__on_disconnected()
            logger.exception("State thread stopped")

    def __run_stream_thread(self):
        reformatter = VideoReformatter()

        try:
            for frame in self.__av_container.decode(video=0):
                if not self.__is_streaming:
                    break
                frame = reformatter.reformat(frame, VIDEO_WIDTH, VIDEO_HEIGHT, "rgb24")
                img = frame.to_ndarray(format="rgb24")
                self.__last_frame = img
                self.__last_frame_time = time()
        except av.error.ExitError:
            logger.exception("Stream ended")

        self.__last_frame = None
        self.__av_container.close()
        del self.__av_container
