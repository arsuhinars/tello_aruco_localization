from threading import Thread
from time import sleep
from typing import cast

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, immvision, implot  # type: ignore

from tello_aruco_nav.common.utils import shift
from tello_aruco_nav.modules.flight_controller import FlightController
from tello_aruco_nav.modules.hud import Hud
from tello_aruco_nav.modules.mission_controller import MissionController
from tello_aruco_nav.modules.tello import Tello, TelloConnectionState
from tello_aruco_nav.modules.tello_controller import TelloController, TelloState

SAMPLES_COUNT = 100
SAMPLE_DELAY = 0.1


class Ui:
    def __init__(
        self,
        tello: Tello,
        controller: TelloController,
        mission_controller: MissionController,
        flight_controller: FlightController,
        hud: Hud,
    ):
        self.__tello = tello
        self.__controller = controller
        self.__mission_controller = mission_controller
        self.__flight_controller = flight_controller
        self.__hud = hud
        self.__is_running = False

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
        self.__params.callbacks.before_exit = self.stop
        self.__params.dpi_aware_params = hello_imgui.DpiAwareParams(1.25)
        self.__params.ini_filename = "app_settings.ini"
        self.__params.fps_idling = hello_imgui.FpsIdling(enable_idling=False)

        self.__pid_x_current = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_x_target = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_x_control = np.zeros(SAMPLES_COUNT, np.float32)

        self.__pid_y_current = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_y_target = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_y_control = np.zeros(SAMPLES_COUNT, np.float32)

        self.__pid_z_current = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_z_target = np.zeros(SAMPLES_COUNT, np.float32)
        self.__pid_z_control = np.zeros(SAMPLES_COUNT, np.float32)

        self.__rate_t = np.arange(-100.0, 100.0, 2.0, np.float32)
        self.__rate_x = np.zeros(100, np.float32)
        self.__rate_y = np.zeros(100, np.float32)
        self.__rate_z = np.zeros(100, np.float32)
        self.__update_rates()

    @property
    def is_running(self):
        return self.__is_running

    def start(self):
        self.__gui_thread = Thread(target=self.__run_gui, daemon=True)
        self.__gui_thread.start()

        self.__sampler_thread = Thread(target=self.__run_sampler, daemon=True)
        self.__sampler_thread.start()

        self.__is_running = True

    def stop(self):
        if self.__is_running:
            del self.__gui_thread
            del self.__sampler_thread

            self.__is_running = False
            self.__params.app_shall_exit = True

    def __run_gui(self):
        immvision.use_rgb_color_order()
        immapp.run(
            self.__params,
            immapp.AddOnsParams(with_implot=True),
        )

    def __run_sampler(self):
        while not self.__params.app_shall_exit:
            self.__pid_x_current = shift(
                self.__pid_x_current, -1, self.__controller.pid_x_state.current
            )

            self.__pid_x_target = shift(
                self.__pid_x_target, -1, self.__controller.pid_x_state.target
            )

            self.__pid_x_control = shift(
                self.__pid_x_control, -1, self.__controller.pid_x_state.control * 0.01
            )

            self.__pid_y_current = shift(
                self.__pid_y_current, -1, self.__controller.pid_y_state.current
            )

            self.__pid_y_target = shift(
                self.__pid_y_target, -1, self.__controller.pid_y_state.target
            )

            self.__pid_y_control = shift(
                self.__pid_y_control, -1, self.__controller.pid_y_state.control * 0.01
            )

            self.__pid_z_current = shift(
                self.__pid_z_current, -1, self.__controller.pid_z_state.current
            )

            self.__pid_z_target = shift(
                self.__pid_z_target, -1, self.__controller.pid_z_state.target
            )

            self.__pid_z_control = shift(
                self.__pid_z_control, -1, self.__controller.pid_z_state.control * 0.01
            )

            sleep(SAMPLE_DELAY)

    def __update_rates(self):
        rate_x, rate_y, rate_z = self.__controller.rate_expo
        for i, t in enumerate(self.__rate_t):
            self.__rate_x[i] = self.__controller.calc_rate_value(cast(float, t), rate_x)
            self.__rate_y[i] = self.__controller.calc_rate_value(cast(float, t), rate_y)
            self.__rate_z[i] = self.__controller.calc_rate_value(cast(float, t), rate_z)

    def __show_status_bar(self):
        imgui.begin_horizontal("status_bar")

        match self.__tello.connection_state:
            case TelloConnectionState.DISCONNECTED:
                imgui.text("Disconnected")
            case TelloConnectionState.CONNECTING:
                imgui.text("Connecting...")
            case TelloConnectionState.CONNECTED:
                imgui.text("Connected.")

                imgui.dummy((20.0, 0.0))
                imgui.text(f"battery {self.__tello.battery}%")

        imgui.end_horizontal()

    def __show_gui(self):
        imgui.dock_space_over_viewport()

        self.__show_pid_x()
        self.__show_pid_y()
        self.__show_pid_z()
        self.__show_rates()
        self.__show_hud()

        self.__flight_controller.trigger_imgui_update()

    def __show_pid_x(self):
        is_shown, _ = imgui.begin("PID X")
        if is_shown:
            if implot.begin_plot("PID x", (-1.0, 400.0)):
                implot.plot_line("current", self.__pid_x_current)
                implot.plot_line("target", self.__pid_x_target)
                implot.plot_line("control", self.__pid_x_control, xscale=0.01)

                implot.end_plot()

            imgui.begin_horizontal("tunings")
            imgui.push_item_width(60.0)
            k_p, k_i, k_d = self.__controller.pid_x

            result_1, k_p = imgui.drag_float("Kp", k_p, 0.1, -100.0, 100.0)
            result_2, k_i = imgui.drag_float("Ki", k_i, 0.1, -100.0, 100.0)
            result_3, k_d = imgui.drag_float("Kd", k_d, 0.1, -100.0, 100.0)

            if result_1 or result_2 or result_3:
                self.__controller.pid_x = k_p, k_i, k_d

            imgui.pop_item_width()
            imgui.end_horizontal()

        imgui.end()

    def __show_pid_y(self):
        is_shown, _ = imgui.begin("PID Y (altitude)")
        if is_shown:
            if implot.begin_plot("PID y", (-1.0, 400.0)):
                implot.plot_line("current", self.__pid_y_current)
                implot.plot_line("target", self.__pid_y_target)
                implot.plot_line("control", self.__pid_y_control)

                implot.end_plot()

            imgui.begin_horizontal("tunings")
            imgui.push_item_width(60.0)
            k_p, k_i, k_d = self.__controller.pid_y

            result_1, k_p = imgui.drag_float("Kp", k_p, 0.1, -100.0, 100.0)
            result_2, k_i = imgui.drag_float("Ki", k_i, 0.1, -100.0, 100.0)
            result_3, k_d = imgui.drag_float("Kd", k_d, 0.1, -100.0, 100.0)

            if result_1 or result_2 or result_3:
                self.__controller.pid_y = k_p, k_i, k_d

            imgui.pop_item_width()
            imgui.end_horizontal()

        imgui.end()

    def __show_pid_z(self):
        is_shown, _ = imgui.begin("PID Z")
        if is_shown:
            if implot.begin_plot("PID z", (-1.0, 400.0)):
                implot.plot_line("current", self.__pid_z_current)
                implot.plot_line("target", self.__pid_z_target)
                implot.plot_line("control", self.__pid_z_control, xscale=0.01)

                implot.end_plot()

            imgui.begin_horizontal("tunings")
            imgui.push_item_width(60.0)
            k_p, k_i, k_d = self.__controller.pid_z

            result_1, k_p = imgui.drag_float("Kp", k_p, 0.1, -100.0, 100.0)
            result_2, k_i = imgui.drag_float("Ki", k_i, 0.1, -100.0, 100.0)
            result_3, k_d = imgui.drag_float("Kd", k_d, 0.1, -100.0, 100.0)

            if result_1 or result_2 or result_3:
                self.__controller.pid_z = k_p, k_i, k_d

            imgui.pop_item_width()
            imgui.end_horizontal()

        imgui.end()

    def __show_rates(self):
        is_shown, _ = imgui.begin("Rates")
        if is_shown:
            if implot.begin_plot("Rates", (-1.0, 400.0)):
                implot.plot_line("left/right", self.__rate_t, self.__rate_x)
                implot.plot_line("up/down", self.__rate_t, self.__rate_y)
                implot.plot_line("forward/backward", self.__rate_t, self.__rate_z)

                implot.end_plot()

            changed, value = imgui.drag_float3(
                "Rates", list(self.__controller.rate_expo), 0.1
            )
            if changed:
                self.__controller.rate_expo = value
                self.__update_rates()

        imgui.end()

    def __show_hud(self):
        from tello_aruco_nav.modules.flight_controller import FlightMode

        is_shown, _ = imgui.begin("HUD")

        if is_shown:
            immvision.image_display(
                "hud_image",
                self.__hud.image,
                refresh_image=True,
            )

            imgui.push_item_width(160.0)
            imgui.begin_horizontal("controls")

            imgui.begin_vertical("left")
            imgui.begin_disabled(
                self.__controller.state in [TelloState.TAKEOFF, TelloState.LANDING]
            )
            match self.__flight_controller.mode:
                case FlightMode.MANUAL | FlightMode.FOLLOW:
                    if self.__controller.state in [TelloState.IDLE, TelloState.TAKEOFF]:
                        if imgui.button("Takeoff"):
                            self.__flight_controller.on_flight_button_clicked()
                    else:
                        if imgui.button("Land"):
                            self.__flight_controller.on_flight_button_clicked()
                case FlightMode.MISSION:
                    if not self.__mission_controller.is_started:
                        if imgui.button("Start mission"):
                            self.__flight_controller.on_flight_button_clicked()
                    else:
                        if imgui.button("Stop mission"):
                            self.__flight_controller.on_flight_button_clicked()
            imgui.end_disabled()

            if imgui.begin_combo("Flight mode", str(self.__flight_controller.mode)):
                for mode in FlightMode:
                    changed, value = imgui.selectable(
                        str(mode), mode == self.__flight_controller.mode
                    )
                    if changed and value:
                        self.__flight_controller.mode = mode
                imgui.end_combo()

            match self.__flight_controller.mode:
                case FlightMode.FOLLOW:
                    marker_id = self.__flight_controller.target_marker_id
                    changed, marker_id = imgui.input_int(
                        "Marker id",
                        marker_id if marker_id is not None else 1,
                        step_fast=0,
                    )
                    if changed:
                        self.__flight_controller.target_marker_id = marker_id

                    changed, altitude = imgui.drag_float(
                        "Altitude",
                        self.__flight_controller.target_altitude,
                        0.025,
                        0.3,
                        3.0,
                    )
                    if changed:
                        self.__flight_controller.target_altitude = altitude

            imgui.end_vertical()

            imgui.pop_item_width()
            imgui.end_horizontal()

        imgui.end()
