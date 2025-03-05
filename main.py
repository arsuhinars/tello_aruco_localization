import logging

from rich.logging import RichHandler

from tello_aruco_nav.common.setup import app
from tello_aruco_nav.common.utils import console

if __name__ == "__main__":
    logging.basicConfig(
        style="{",
        format="{name}: {message}",
        datefmt="%X",
        handlers=[RichHandler(console=console)],
    )

    app()
