from threading import Thread
from time import sleep

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot  # type: ignore

from tello_aruco_nav.common.utils import shift
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState

SAMPLES_COUNT = 100
SAMPLE_DELAY = 1.0 / 30.0
FRAMERATE = 60.0


class Plotter:
    def __init__(self, tello: Tello, controller: TelloController):
        self.__tello = tello
        self.__controller = controller

        self.__params = hello_imgui.RunnerParams()

        self.__params.app_window_params.window_title = "tello_aruco_nav"
        self.__params.app_window_params.window_geometry = hello_imgui.WindowGeometry(
            window_size_state=hello_imgui.WindowSizeState.maximized,
        )

        self.__params.imgui_window_params.show_status_bar = True
        self.__params.callbacks.show_status = self.__show_status_bar

        self.__params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.no_default_window
        )
        self.__params.callbacks.show_gui = self.__show_gui
        self.__params.dpi_aware_params = hello_imgui.DpiAwareParams(1.25)
        self.__params.ini_filename = ""
        self.__params.fps_idling = hello_imgui.FpsIdling(
            enable_idling=False,
        )

        # self.__pid_time = np.zeros(SAMPLES_COUNT)
        self.__pid_y_current = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_y_target = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_y_control = np.zeros(SAMPLES_COUNT, np.float32)

    def start(self):
        self.__gui_thread = Thread(target=self.__run_gui, daemon=True)
        self.__gui_thread.start()

        self.__sampler_thread = Thread(target=self.__run_sampler, daemon=True)
        self.__sampler_thread.start()

    def stop(self):
        del self.__gui_thread
        del self.__sampler_thread

        self.__params.app_shall_exit = True

    def __run_gui(self):
        immapp.run(
            self.__params,
            immapp.AddOnsParams(with_implot=True),
        )

    def __run_sampler(self):
        while not self.__params.app_shall_exit:
            # self.__pid_time = shift(self.__pid_time, -1)
            # self.__pid_time[-1] = time()

            self.__pid_y_current = shift(self.__pid_y_current, -1)
            self.__pid_y_current[-1] = self.__controller.pid_y_state.current

            self.__pid_y_target = shift(self.__pid_y_target, -1)
            self.__pid_y_target[-1] = self.__controller.pid_y_state.target

            self.__pid_y_control = shift(self.__pid_y_control, -1)
            self.__pid_y_control[-1] = self.__controller.pid_y_state.control

            sleep(SAMPLE_DELAY)

    def __show_status_bar(self):
        imgui.begin_horizontal("status_bar")

        match self.__tello.connection_state:
            case TelloConnectionState.DISCONNECTED:
                imgui.text("Disconnected")
            case TelloConnectionState.CONNECTING:
                imgui.text("Connecting...")
            case TelloConnectionState.CONNECTED:
                imgui.text("Connected")

                imgui.dummy((20.0, 0.0))
                imgui.text(f"bat={self.__tello.battery}%, h={self.__tello.height} m.")
                imgui.dummy((20.0, 0.0))

                match self.__controller.state:
                    case TelloState.IDLE:
                        imgui.text("idle")
                    case TelloState.TAKEOFF:
                        imgui.text("takeoff")
                    case TelloState.LANDING:
                        imgui.text("landing")
                    case TelloState.MANUAL_CONTROL:
                        if self.__controller.manual_control is not None:
                            x, y, z, w = self.__controller.manual_control
                            imgui.text(
                                f"manual control: left_right={x}, forward_backward={y}, up_down={z}, yaw={w})"
                            )
                    case TelloState.GO_TO_MARKER:
                        marker_id = self.__controller.marker_id
                        marker_dist = self.__controller.marker_dist
                        marker_alt_delta = self.__controller.marker_alt_delta
                        if marker_dist is not None:
                            imgui.text(
                                f"going to marker: id={marker_id}, dist={marker_dist:.2f} m., alt_delta={marker_alt_delta:.2f} m."
                            )

        imgui.end_horizontal()

    def __show_gui(self):
        imgui.dock_space_over_viewport()

        imgui.begin("PID Y")
        if implot.begin_plot("PID y", (-1.0, 400.0)):
            implot.plot_line("current", self.__pid_y_current)
            implot.plot_line("target", self.__pid_y_target)
            implot.plot_line("control", self.__pid_y_control, xscale=0.01)

            implot.end_plot()

        imgui.begin_horizontal("tunings")
        imgui.push_item_width(60.0)
        k_p, k_i, k_d = self.__controller.pid_y

        _, k_p = imgui.drag_float("Kp", k_p, 0.1, -100.0, 100.0)
        _, k_i = imgui.drag_float("Ki", k_i, 0.1, -100.0, 100.0)
        _, k_d = imgui.drag_float("Kd", k_d, 0.1, -100.0, 100.0)
        _, alt = imgui.drag_float(
            "target", self.__controller.target_altitude, 0.1, 0.2, 3.0
        )

        self.__controller.pid_y = k_p, k_i, k_d
        self.__controller.target_altitude = alt

        imgui.pop_item_width()
        imgui.end_horizontal()

        imgui.end()
