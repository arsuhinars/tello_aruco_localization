class AppException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class CvCameraInitException(AppException):
    def __init__(self):
        super().__init__("Failed to initialize OpenCV camera")


class CvCameraReadFailedException(AppException):
    def __init__(self):
        super().__init__("Failed to read next frame")
