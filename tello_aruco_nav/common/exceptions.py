class AppException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class CvCameraInitException(AppException):
    def __init__(self):
        super().__init__("Failed to initialize OpenCV camera")


class CvCameraReadFailedException(AppException):
    def __init__(self):
        super().__init__("Failed to read next frame")


class TelloAlreadyConnectedException(AppException):
    def __init__(self):
        super().__init__("Tello is already connected")


class TelloBusyException(AppException):
    def __init__(self):
        super().__init__(
            "Unable to send command, because waiting for response from another one"
        )


class TelloErrorException(AppException):
    def __init__(self):
        super().__init__("Tello returned error response")


class TelloFailedConnectException(AppException):
    def __init__(self):
        super().__init__("Failed to connected to Tello")


class TelloDisconnectedException(AppException):
    def __init__(self):
        super().__init__("Tello disconnected")
