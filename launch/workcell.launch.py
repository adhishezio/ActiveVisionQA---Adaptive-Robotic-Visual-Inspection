"""UR5e, actual Gazebo camera and MoveIt, using the same generated workcell."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder

from activevisionqa.assets import WORLD_SCENARIOS, generate_workcell
from activevisionqa.config import Config


def launch_setup(context):
    share = Path(get_package_share_directory("activevisionqa"))
    ur_sim = Path(get_package_share_directory("ur_simulation_gz"))
    scenario = LaunchConfiguration("scenario").perform(context)
    config_path = LaunchConfiguration("config").perform(context)
    config = Config.load(config_path)
    if config.world_frame != "world" or config.optical_frame != "inspection_camera_optical_frame":
        raise ValueError("The shipped URDF requires world/inspection_camera_optical_frame")
    root = Path(LaunchConfiguration("output_root").perform(context)).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    directory = root / f"world_{scenario}_{run_id}"
    world = generate_workcell(directory, scenario, config)
    metadata = {"backend": "gazebo", "scenario": scenario, "robot_model": "ur5e",
                "config": config.to_dict(), "world_file": str(world), "simulation_only": True}
    (directory / "world_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    camera = share / "urdf/inspection_camera.urdf.xacro"
    controllers = ur_sim / "config/ur_controllers.yaml"
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(ur_sim / "launch/ur_sim_control.launch.py")),
        launch_arguments={"ur_type": "ur5e", "description_file": str(camera),
                          "world_file": str(world), "gazebo_gui": LaunchConfiguration("gui"),
                          "launch_rviz": "false", "safety_limits": "true",
                          "controllers_file": str(controllers),
                          "initial_joint_controller": "joint_trajectory_controller"}.items(),
    )
    moveit = (MoveItConfigsBuilder("ur", package_name="ur_moveit_config")
              .robot_description(file_path=str(camera), mappings={"ur_type": "ur5e", "name": "ur",
                                  "simulation_controllers": str(controllers), "safety_limits": "true"})
              .robot_description_semantic(file_path="srdf/ur.srdf.xacro", mappings={"name": "ur"})
              .planning_pipelines(pipelines=["ompl"])
              .trajectory_execution(file_path="config/moveit_controllers.yaml")
              .to_moveit_configs())
    execution = moveit.trajectory_execution["moveit_simple_controller_manager"]
    execution["controller_names"] = ["joint_trajectory_controller"]
    execution["joint_trajectory_controller"]["default"] = True
    execution["scaled_joint_trajectory_controller"]["default"] = False
    moveit_parameters = moveit.to_dict()
    moveit_parameters["robot_description"] = ParameterValue(moveit.robot_description["robot_description"], value_type=str)
    moveit_parameters["robot_description_semantic"] = ParameterValue(moveit.robot_description_semantic["robot_description_semantic"], value_type=str)
    move_group = Node(
        package="moveit_ros_move_group", executable="move_group", output="screen",
        parameters=[moveit_parameters, {"use_sim_time": True,
                    "publish_robot_description_semantic": True,
                    "trajectory_execution.allowed_start_tolerance": 0.03}],
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="inspection_camera_bridge",
        output="screen", arguments=[
            config.camera_topic+"@sensor_msgs/msg/Image[gz.msgs.Image",
            config.camera_info_topic+"@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"],
        parameters=[{"use_sim_time": True, "override_frame_id": config.optical_frame}],
    )
    information = Node(package="activevisionqa", executable="world_metadata", output="screen",
                       parameters=[{"use_sim_time": True, "metadata_json": ParameterValue(json.dumps(metadata), value_type=str)}])
    print(f"ActiveVisionQA workcell: {directory}", flush=True)
    return [simulation, move_group, bridge, information]


def generate_launch_description():
    share = Path(get_package_share_directory("activevisionqa"))
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="occluded", choices=WORLD_SCENARIOS),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("config", default_value=str(share / "config/inspection.json")),
        DeclareLaunchArgument("output_root", default_value=os.path.join(os.getcwd(), "outputs")),
        OpaqueFunction(function=launch_setup),
    ])
