import typer

from tello_aruco_nav.runners.calibration import calibrate_camera
from tello_aruco_nav.runners.flight import run_flight
from tello_aruco_nav.runners.markers_generation import generate_markers

app = typer.Typer()

app.command()(generate_markers)
app.command()(calibrate_camera)
app.command()(run_flight)
