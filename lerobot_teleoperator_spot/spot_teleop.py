import logging
import time
import socket
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

from lerobot.teleoperators.teleoperator import Teleoperator

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME,ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.inverse_kinematics import InverseKinematicsClient
from bosdyn.api.spot.inverse_kinematics_pb2 import (InverseKinematicsRequest,
                                                    InverseKinematicsResponse)
from bosdyn.client.math_helpers import SE3Pose, Quat

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
        # arm movement
        bosdyn.client.util.setup_logging()
        self._sdk = bosdyn.client.create_standard_sdk('Spot_LeRobot_Robot')
        self.robot = self._sdk.create_robot(self.config.robot_ip)
        self.robot.authenticate(self.config.robot_user,
                                  self.config.robot_password)
        bosdyn.client.util.authenticate(self.robot)
        self.robot_state_client: RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.ik_client: InverseKinematicsClient = self.robot.ensure_client(InverseKinematicsClient.default_service_name)
        self.robot_state = self.robot_state_client.get_robot_state()
        ## vars for arm movement
        self.pos_when_triggered = np.array([0.0, 0.0, 0.0])
        self.rot_when_triggered = R.from_quat([0.0, 0.0, 0.0, 1.0])
        self.robot_pos_at_trigger = None
        self.robot_rot_at_trigger = None
        self.button_already_pressed = False
        self.rot_offset = None
        ## vars for gripper
        self.gripper_pos = 0.0 
        self.gripper_speed = 0.02


    @property
    def action_features(self) -> dict[str, type]:
        return {
            "base.x.vel": float,
            "base.x.vel": float,
            "base.rot.vel": float,            
            "arm.x": float,
            "arm.y": float,
            "arm.z": float,
            "arm.rot.w": float,
            "arm.rot.x": float,
            "arm.rot.y": float,
            "arm.rot.z": float,
            "gripper.pos": float,
            "arm_control": bool,
            "arm_carry_enabled": bool,
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

        self.robot_state = self.robot_state_client.get_robot_state()
        body_tform_hand = get_a_tform_b(
            self.robot_state.kinematic_state.transforms_snapshot,
            GRAV_ALIGNED_BODY_FRAME_NAME,
            HAND_FRAME_NAME
        )
        curr_hand_quat = [body_tform_hand.rot.x, body_tform_hand.rot.y, body_tform_hand.rot.z, body_tform_hand.rot.w]

        action_dict = {
            "base.x.vel": 0.0,
            "base.y.vel": 0.0,
            "base.rot.vel": 0.0,
            "arm_control": False,
            "arm_carry_enabled": False,
            "arm.x": body_tform_hand.x, # Default reach
            "arm.y": body_tform_hand.y,
            "arm.z": body_tform_hand.z,
            "arm.rot.w": body_tform_hand.rot.w,
            "arm.rot.x": body_tform_hand.rot.x,
            "arm.rot.y": body_tform_hand.rot.y,
            "arm.rot.z": body_tform_hand.rot.z,
            "gripper.pos": self.gripper_pos,
        }

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
            stick = formated_data_left.get('stick', [0, 0])
            # x is Godot Y
            action_dict["base.x.vel"] = stick[1] # Forward/Backward
            if abs(stick[0]) > 0.5:
                action_dict["base.y.vel"] = -0.5 * stick[0]
        
        # base rotation movement
        if formated_data_right is not None:
            stick_right = formated_data_right.get('stick', [0, 0])
            action_dict["base.rot.vel"] = stick_right[1]

            if formated_data_right["btn_by"]:
                action_dict['arm_carry_enabled'] = True

            # arm movement
            if formated_data_right["btn_ax"]:
                # get controller pos
                curr_pos = np.array(formated_data_right["pos"])
                curr_quat = np.array(formated_data_right["quat"])

                curr_vr_rot_raw = R.from_quat(curr_quat)
                vr_rot_vec = curr_vr_rot_raw.as_rotvec()
                remap_vr_curr_vec = np.array([
                    -vr_rot_vec[2], # Spot X
                    -vr_rot_vec[0], # Spot Y
                    vr_rot_vec[1]  # Spot Z
                ])
                remap_vr_curr_obj = R.from_rotvec(remap_vr_curr_vec) 
                
                # check if button pressed
                if not self.button_already_pressed:
                    # safe controller and robot pos as reference
                    self.pos_when_triggered = curr_pos
                    self.robot_rot_at_trigger = R.from_quat(curr_hand_quat)
                    self.robot_pos_at_trigger = np.array([body_tform_hand.x, body_tform_hand.y, body_tform_hand.z])
                    # save controller and robot offset to keep control axis
                    # when the arm is rotated
                    self.rot_offset = self.robot_rot_at_trigger * remap_vr_curr_obj.inv()                    
                    # button state
                    self.button_already_pressed = True
                else:
                    # calcuate movement based on the reference
                    delta_pos_abs = curr_pos - self.pos_when_triggered
                    remap_pos_abs = np.array([-delta_pos_abs[2], -delta_pos_abs[0], delta_pos_abs[1]])
                    target_pos_abs = self.robot_pos_at_trigger + remap_pos_abs
                    target_rot_obj = self.rot_offset * remap_vr_curr_obj
                    target_quat = target_rot_obj.as_quat() # returns [x, y, z, w]

                    ik_joints = self.get_joints_from_pose(
                        target_pos_abs[0], target_pos_abs[1], target_pos_abs[2],
                        target_quat[3], target_quat[0], target_quat[1], target_quat[2]
                    )
                    if ik_joints:
                        for i, val in enumerate(ik_joints):
                            action_dict[f"arm.joint{i+1}.pos"] = val

                    action_dict["arm_control"] = True
                    action_dict["arm.x"] = target_pos_abs[0]
                    action_dict["arm.y"] = target_pos_abs[1]
                    action_dict["arm.z"] = target_pos_abs[2]
                    
                    action_dict["arm.rot.x"] = target_quat[0]
                    action_dict["arm.rot.y"] = target_quat[1]
                    action_dict["arm.rot.z"] = target_quat[2]
                    action_dict["arm.rot.w"] = target_quat[3]       
            else:
                # reset button state
                self.button_already_pressed = False

            if formated_data_right['trigger']:
                # close gripper step wise
                self.gripper_pos -= self.gripper_speed
            
            # open gripper
            if formated_data_right['grip']:
                # open gripper stepwise
                self.gripper_pos += self.gripper_speed

            self.gripper_pos = float(np.clip(self.gripper_pos, 0.0, 1.0))
            action_dict["gripper.pos"] = self.gripper_pos

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
    
    def get_joints_from_pose(self, x, y, z, rw, rx, ry, rz):
        snapshot = self.robot_state.kinematic_state.transforms_snapshot
        odom_T_body = get_a_tform_b(snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # 2. Define the target tool pose relative to the BODY (from your VR math)
        body_T_tool = SE3Pose(x, y, z, Quat(w=rw, x=rx, y=ry, z=rz))

        # 3. Transform the target tool pose into the ODOM frame
        odom_T_tool = odom_T_body * body_T_tool

        # 4. Build the request with ODOM as the root
        request = InverseKinematicsRequest(
            root_frame_name=ODOM_FRAME_NAME,
            tool_pose_task=InverseKinematicsRequest.ToolPoseTask(
                task_tform_desired_tool=odom_T_tool.to_proto()
            )
        )
        
        # Call the service
        try:
            response = self.ik_client.inverse_kinematics(request)
            
            if response.status == InverseKinematicsResponse.STATUS_OK:
                joint_states = response.robot_configuration.joint_states
    
                # 1. Filter for all arm-related joints
                arm_proto_joints = [j for j in joint_states if j.name.startswith("arm0")]
                
                # 2. Separate the 6 arm joints from the 1 gripper joint (f1x)
                # This matches your logic: sorted(arm0 joints NOT ending in f1x)
                arm_joints_only = sorted(
                    [j for j in arm_proto_joints if not j.name.endswith("f1x")],
                    key=lambda x: x.name
                )
                
                # 3. Extract the numeric values (.value is required for Protobuf wrappers)
                arm_joint_values = [j.position.value for j in arm_joints_only]
                
                # Return exactly the 6 arm joint angles
                return arm_joint_values
            else:
                self.robot.logger.error(f"IK solver failed with status: {response.status}")
                return None
        except Exception as e:
            self.robot.logger.error(f"IK request failed: {e}")
            return None

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

    #@check_if_not_connected
    def disconnect(self) -> None:
        self.sock_left.close()
        self.sock_right.close()
        logger.info(f"{self} disconnected.")
        
