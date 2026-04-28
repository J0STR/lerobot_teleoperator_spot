import logging
import time
import socket
import numpy as np
import json

from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)

from .config_spot_teleop import SpotTeleopConfig

### for reading controller msges
def process_controller_data(data_bytes: bytes):
        try:
            msg = json.loads(data_bytes.decode('utf-8'))
            return msg
        except Exception as e:
            print(f"JSON Error: {e}")

class SpotTeleop(Teleoperator):
    config_class = SpotTeleopConfig
    name = "Spot Teleoperator"

    def __init__(self, config: SpotTeleopConfig):
        super().__init__(config)
        self.config = config
        # left controller socket
        self.sock_left = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_left.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_left.bind((self.config.local_host, int(self.config.port_controller_left)))
        self.sock_left.setblocking(False)
        # right controller socket
        self.sock_right = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_right.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_right.bind((self.config.local_host, int(self.config.port_controller_right)))
        self.sock_right.setblocking(False)


    @property
    def action_features(self) -> dict[str, type]:
        return {
            "x_axis.vel": float,
            "y_axis.vel": float,
            "rotation.vel": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return True    

    #@check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self) -> None:
        logger.info(f"{self} calibrated.")

    def configure(self) -> None:
        logger.info(f"{self} configured.")

    def setup_motors(self) -> None:
        logger.info(f"{self} motors setup.")

    #@check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()

        action_dict = {}
        action = np.array([0.,0.,0.])
        action_dict["x_axis.vel"] = action[0]
        action_dict["y_axis.vel"] = action[1]
        action_dict["rotation.vel"] = action[2]

        # Drain buffers to get latest data
        latest_data_bytes_left , latest_data_bytes_right = self.drain_buffers()
        # process left data
        if latest_data_bytes_left is not None:
            formated_data_left = process_controller_data(latest_data_bytes_left) 
        else:
            formated_data_left = None
        # process right data
        if latest_data_bytes_right is not None:
            formated_data_right = process_controller_data(latest_data_bytes_right) 
        else:
            formated_data_right = None
        # Check data
        if formated_data_left is None and formated_data_right is None:
            return action_dict

        # process data and get action 
        if formated_data_left is not None:
            linear_movement = formated_data_left['stick']
            action[0]= linear_movement[1]
            if abs(linear_movement[0])>0.5:
                action[1]= -linear_movement[0]
            action_dict["x_axis.vel"] = action[0]
            action_dict["y_axis.vel"] = action[1]
        if formated_data_right is not None:
            rotation = formated_data_right['stick']
            action[2] = rotation[1]
            action_dict["rotation.vel"] = action[2]     

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action_dict
    
    def drain_buffers(self):
        latest_data_bytes_left = None
        while True:
            try:
                data, _ = self.sock_left.recvfrom(4096)
                latest_data_bytes_left = data
            except BlockingIOError:
                break
        latest_data_bytes_right = None
        while True:
            try:
                data, _ = self.sock_right.recvfrom(4096)
                latest_data_bytes_right = data
            except BlockingIOError:
                break

        return latest_data_bytes_left, latest_data_bytes_right

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

    #@check_if_not_connected
    def disconnect(self) -> None:
        self.sock_left.close()
        self.sock_right.close()
        logger.info(f"{self} disconnected.")
        
