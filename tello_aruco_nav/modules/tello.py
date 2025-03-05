import logging
import re
import socket
from collections import defaultdict
from enum import IntEnum
from select import select
from threading import Thread
from time import time

import av
import av.error
import numpy as np

from tello_aruco_nav.common.exceptions import (
    TelloBusyException,
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
MAX_RETRIES = 3
TELLO_STATE_REGEX = re.compile(r"(\w+):(\d*(?:.\d{1,2})?)(?:;|$)")


class TelloConnectionState(IntEnum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2


class Tello:
    def __init__(self):
        self.__connection_state = TelloConnectionState.DISCONNECTED
        self.__is_streaming = False
        self.__is_busy = False
        self.__is_flying = False
        self.__retries_cnt = 0
        self.__last_response_time = 0.0
        self.__last_frame_time = 0.0
        self.__last_sent_message: bytes | None = None
        self.__last_frame: np.ndarray | None = None
        self.__state_dict: dict[str, float] = defaultdict(lambda: 0.0)

    @property
    def connection_state(self):
        return self.__connection_state

    @property
    def is_streaming(self):
        return self.__is_streaming

    @property
    def is_busy(self):
        return self.__is_busy

    @property
    def is_flying(self):
        return self.__is_flying

    @property
    def battery(self):
        return int(self.__state_dict["bat"])

    def connect(self):
        self.__cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.__client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__client_sock.bind(("0.0.0.0", TELLO_COMMAND_PORT))
        self.__client_sock.setblocking(False)

        self.__state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__state_sock.bind(("0.0.0.0", TELLO_STATE_PORT))
        self.__state_sock.setblocking(False)

        self.__av_container.close()

        try:
            select([], [self.__cmd_sock], [])
            self.__cmd_sock.sendto(b"command", TELLO_ADDRESS)
        except OSError:
            raise TelloFailedConnectException()

        self.__connection_state = TelloConnectionState.CONNECTING
        self.__is_streaming = False
        self.__is_busy = True
        self.__is_flying = False
        self.__last_response_time = time()
        self.__last_frame_time = time()

        logger.info("Connecting")

    def disconnect(self):
        try:
            self.__cmd_sock.sendto(b"emergency", TELLO_ADDRESS)
        except OSError:
            logger.exception("Failed to send emergency message during disconnecting")

        self.__cmd_sock.close()
        self.__client_sock.close()
        self.__state_sock.close()

        del self.__cmd_sock
        del self.__client_sock
        del self.__state_sock
        del self.__av_thread

        self.__connection_state = TelloConnectionState.DISCONNECTED
        self.__is_streaming = False
        self.__is_busy = False
        self.__is_flying = False

        logger.info("Disconnected")

    def stream_on(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command(b"streamon")
        self.__av_container = av.open(
            f"udp://0.0.0.0:{TELLO_STREAM_PORT}",
            timeout=TIMEOUT_DELAY,
        )
        self.__av_thread = Thread(target=self.__run_stream_thread)
        self.__av_thread.run()
        self.__is_streaming = True

    def stream_off(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command(b"streamoff")

        del self.__av_thread

        self.__av_container.close()
        self.__is_streaming = False

    def takeoff(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command(b"takeoff")
        self.__is_flying = True

    def land(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command(b"land")
        self.__is_flying = False

    def emergency(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command(b"emergency")
        self.__is_flying = False

    def send_rc_control(
        self, left_right: int, forward_backward: int, up_down: int, yaw: int
    ):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        self.__send_command_no_response(
            f"rc {left_right} {forward_backward} {up_down} {yaw}".encode("ascii")
        )

    def read_next_frame(self):
        if self.__connection_state != TelloConnectionState.CONNECTED:
            raise TelloDisconnectedException()

        if time() - self.__last_frame_time > TIMEOUT_DELAY:
            return None

        return self.__last_frame

    def update(self):
        if self.__connection_state == TelloConnectionState.DISCONNECTED:
            return

        if time() - self.__last_response_time > TIMEOUT_DELAY:
            if (
                self.__last_sent_message is not None
                and self.__retries_cnt < MAX_RETRIES
            ):
                select([], [self.__cmd_sock], [])
                self.__cmd_sock.sendto(self.__last_sent_message, TELLO_ADDRESS)
                self.__last_response_time = time()
                self.__retries_cnt += 1
                logger.warning(
                    f"Response timeout exceeded. Retrying ({self.__retries_cnt} of {MAX_RETRIES})"
                )
            elif self.__connection_state == TelloConnectionState.CONNECTING:
                logger.exception("Failed to connect")
                raise TelloFailedConnectException()
            else:
                logger.exception("Response timeout exceeded. Disconnected")
                raise TelloDisconnectedException()

        read_socks, _, _ = select([self.__client_sock, self.__state_sock], [], [], 0.0)

        if self.__client_sock in read_socks:
            match self.__client_sock.recv(256):
                case b"ok":
                    if self.__connection_state == TelloConnectionState.CONNECTING:
                        self.__connection_state = TelloConnectionState.CONNECTED
                        logger.info("Connected")
                    self.__is_busy = False
                    self.__last_sent_message = None
                    self.__last_response_time = time()
                    self.__retries_cnt = 0
                case b"error":
                    raise TelloErrorException()

        if self.__state_sock in read_socks:
            state_msg = self.__state_sock.recv(1024).decode("ascii")
            for m in TELLO_STATE_REGEX.finditer(state_msg):
                self.__state_dict[m.group(1)] = float(m.group(2))

    def __send_command(self, command: bytes):
        if self.__is_busy:
            raise TelloBusyException()

        select([], [self.__cmd_sock], [])

        self.__last_sent_message = command
        self.__cmd_sock.sendto(command, TELLO_ADDRESS)
        self.__is_busy = True

    def __send_command_no_response(self, command: bytes):
        self.__cmd_sock.sendto(command, TELLO_ADDRESS)

    def __run_stream_thread(self):
        try:
            for frame in self.__av_container.decode(video=0):
                if not self.__is_streaming:
                    self.__av_container.close()
                    break

                img = frame.to_ndarray(format="rgb24")
                self.__last_frame = img
                self.__last_frame_time = time()
        except av.error.ExitError:
            logger.exception("Stream ended")
