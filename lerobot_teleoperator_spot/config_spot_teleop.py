from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

from .login_data import * # add your own info here

@TeleoperatorConfig.register_subclass("SpotTeleop")
@dataclass
class SpotTeleopConfig(TeleoperatorConfig):

    port_controller_left:   str= "5003"
    port_controller_right:  str= "5002"
    local_host: str = "127.0.0.1"

    robot_ip: str = "192.168.80.3"
    robot_user: str = user_name
    robot_password: str = user_password

    # Whether to use degrees for angles
    use_degrees: bool = True