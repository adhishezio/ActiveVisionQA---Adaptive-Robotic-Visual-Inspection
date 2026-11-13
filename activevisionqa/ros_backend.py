"""Gazebo camera and MoveIt action adapter. No real-hardware launch path."""

from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from .assets import scene_boxes
from .config import Config, POLICY_FIELDS
from .geometry import pose_error, pose_from_matrix, pose_matrix
from .models import AcquisitionError, InspectionResult, MotionResult, Observation, Outcome, Pose, SceneBox, ViewContext, Viewpoint
from .perception import MarkerInspector
from .policy import InspectionRunner
from .recording import RunRecorder
from .report import render_report

UR_ARM_JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def transform_pose(transform, reference_frame: str) -> Pose:
    value = transform.transform
    return Pose((value.translation.x, value.translation.y, value.translation.z),
                (value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w), reference_frame)


class RosBackend:
    """Plan, check, execute, verify pose, then acquire a timestamped real sim image."""

    mode = "gazebo_moveit"

    def __init__(self, config: Config, scenario: str):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
        from cv_bridge import CvBridge
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from control_msgs.action import FollowJointTrajectory
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionIK, GetStateValidity
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformListener
        from rclpy.duration import Duration

        self.rclpy, self.config, self.scenario = rclpy, config, scenario
        self.lock = threading.Lock()
        self.image_message = self.camera_info = self.joint_state = None
        self.metadata = None
        self.description = None
        self.clock = 0.0
        self.clock_received = 0.0
        self.joint_limits: dict[str, tuple[float, float, float]] = {}
        self.bridge = CvBridge()
        self.node = Node("activevisionqa_inspection", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.group = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(Image, config.camera_topic, self._image, qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, config.camera_info_topic, self._info, qos_profile_sensor_data)
        self.node.create_subscription(JointState, "/joint_states", self._joints, qos_profile_sensor_data)
        self.node.create_subscription(Clock, "/clock", self._clock, qos_profile_sensor_data)
        self.node.create_subscription(String, "/robot_description", self._description, latched)
        self.node.create_subscription(String, "/activevisionqa/workcell", self._metadata, latched)
        self.tf = Buffer(cache_time=Duration(seconds=30.0), node=self.node)
        self.listener = TransformListener(self.tf, self.node, spin_thread=False)
        self.plan_client = ActionClient(self.node, MoveGroup, "/move_action", callback_group=self.group)
        self.execute_client = ActionClient(self.node, ExecuteTrajectory, "/execute_trajectory", callback_group=self.group)
        # Readiness only: execution still goes through MoveIt. Joint-state
        # publication can begin before the trajectory controller is active.
        self.controller_client = ActionClient(self.node, FollowJointTrajectory,
                                             "/joint_trajectory_controller/follow_joint_trajectory", callback_group=self.group)
        self.apply_client = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=self.group)
        self.scene_client = self.node.create_client(GetPlanningScene, "/get_planning_scene", callback_group=self.group)
        self.validity_client = self.node.create_client(GetStateValidity, "/check_state_validity", callback_group=self.group)
        self.ik_client = self.node.create_client(GetPositionIK, "/compute_ik", callback_group=self.group)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        def spin():
            from rclpy.executors import ExternalShutdownException, ShutdownException
            try:
                self.executor.spin()
            except (ExternalShutdownException, ShutdownException):
                pass
        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()
        try:
            self._ready()
            self._apply_scene()
        except Exception:
            self.close()
            raise

    def _image(self, message):
        with self.lock:
            self.image_message = message

    def _info(self, message):
        with self.lock:
            self.camera_info = message

    def _joints(self, message):
        with self.lock:
            self.joint_state = message

    def _clock(self, message):
        with self.lock:
            self.clock = stamp_seconds(message.clock)
            self.clock_received = time.monotonic()

    def _description(self, message):
        self.description = message.data

    def _metadata(self, message):
        try:
            self.metadata = json.loads(message.data)
        except (ValueError, TypeError):
            self.metadata = None

    def _wait(self, future, timeout: float):
        end = time.monotonic() + timeout
        while not future.done() and self.rclpy.ok():
            if time.monotonic() >= end:
                raise TimeoutError("ROS operation timed out")
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError("ROS shutdown during operation")
        return future.result()

    def _ready(self) -> None:
        end = time.monotonic() + 90.0
        while self.rclpy.ok() and time.monotonic() < end:
            if (self.description and self.metadata and self.joint_state is not None and
                    self.camera_info is not None and self.image_message is not None and self.clock > 0):
                break
            time.sleep(0.1)
        else:
            missing = [name for name, ready in (
                ("robot_description", bool(self.description)), ("workcell metadata", bool(self.metadata)),
                ("joint states", self.joint_state is not None), ("camera info", self.camera_info is not None),
                ("camera images", self.image_message is not None), ("simulation clock", self.clock > 0)) if not ready]
            raise RuntimeError("Simulation not ready: " + ", ".join(missing))
        if not self.node.get_parameter("use_sim_time").value or not self.metadata.get("simulation_only"):
            raise RuntimeError("This adapter requires the supplied simulation-only workcell")
        if self.metadata.get("scenario") != self.scenario:
            raise RuntimeError("Requested scenario does not match running workcell")
        # Fill newly added defaults when connecting to an older compatible world.
        actual = Config.from_dict(self.metadata.get("config", {})).to_dict()
        expected = self.config.to_dict()
        for key in POLICY_FIELDS:
            actual.pop(key, None)
            expected.pop(key, None)
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise RuntimeError("Inspector configuration differs from the generated workcell snapshot")
        info = self.camera_info
        if info.header.frame_id != self.config.optical_frame or info.k[0] <= 0 or info.k[4] <= 0:
            raise RuntimeError("CameraInfo optical frame or intrinsics are invalid")
        if self.image_message.width != info.width or self.image_message.height != info.height:
            raise RuntimeError("Image and CameraInfo resolutions differ")
        root = ET.fromstring(self.description)
        if root.find(".//hardware/plugin") is None:
            raise RuntimeError("Robot description has no simulation control hardware")
        plugins = [element.text or "" for element in root.findall(".//hardware/plugin")]
        if not all("gz_ros2_control" in value for value in plugins):
            raise RuntimeError("Non-Gazebo hardware plugin rejected by simulation adapter")
        for joint in root.findall("joint"):
            if joint.get("type") != "revolute":
                continue
            limits = joint.find("limit")
            if limits is None:
                raise RuntimeError("Revolute joint has no explicit limits")
            low, high, velocity = float(limits.get("lower")), float(limits.get("upper")), float(limits.get("velocity"))
            soft = joint.find("safety_controller")
            if soft is not None:
                low = max(low, float(soft.get("soft_lower_limit", low)))
                high = min(high, float(soft.get("soft_upper_limit", high)))
            self.joint_limits[joint.get("name")] = (low, high, velocity)
        if len(self.joint_limits) != 6:
            raise RuntimeError("Expected the six explicitly bounded UR arm joints")
        for client in (self.plan_client, self.execute_client, self.controller_client):
            if not client.wait_for_server(timeout_sec=15.0):
                raise RuntimeError("MoveIt or trajectory-controller action server unavailable")
        for client in (self.apply_client, self.scene_client, self.validity_client, self.ik_client):
            if not client.wait_for_service(timeout_sec=15.0):
                raise RuntimeError("MoveIt planning-scene service unavailable")
        self._lookup(self.config.tool_frame, self.config.optical_frame)

    def _lookup(self, reference: str, child: str, stamp=None) -> Pose:
        from rclpy.time import Time
        query = Time() if stamp is None else Time.from_msg(stamp)
        transform = self.tf.lookup_transform(reference, child, query)
        return transform_pose(transform, reference)

    def _apply_scene(self) -> None:
        from geometry_msgs.msg import Pose as RosPose
        from moveit_msgs.msg import AllowedCollisionEntry, CollisionObject, PlanningScene, PlanningSceneComponents
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        from shape_msgs.msg import SolidPrimitive
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        matrix = self._wait(self.scene_client.call_async(request), 10).scene.allowed_collision_matrix
        root = ET.fromstring(self.description)
        links = {element.get("name") for element in root.findall("link")}
        for name in ("inspection_floor", "base_link", "base_link_inertia"):
            if name not in links and name != "inspection_floor":
                continue
            if name not in matrix.entry_names:
                for entry in matrix.entry_values:
                    entry.enabled.append(False)
                matrix.entry_names.append(name)
                matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(matrix.entry_names)))
        floor = matrix.entry_names.index("inspection_floor")
        for base in ("base_link", "base_link_inertia"):
            if base in matrix.entry_names:
                index = matrix.entry_names.index(base)
                matrix.entry_values[floor].enabled[index] = True
                matrix.entry_values[index].enabled[floor] = True
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = matrix
        boxes = [{"name": "inspection_floor", "size": [4., 4., .02], "position": [0., 0., -.01]}]
        boxes += scene_boxes(self.scenario, self.config)
        for box in boxes:
            obj = CollisionObject()
            obj.header.frame_id = self.config.world_frame
            obj.pose.orientation.w = 1.0
            obj.id = box["name"]
            shape = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in box["size"]])
            pose = RosPose()
            pose.position.x, pose.position.y, pose.position.z = map(float, box["position"])
            pose.orientation.w = 1.0
            obj.primitives, obj.primitive_poses = [shape], [pose]
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)
        apply = ApplyPlanningScene.Request(scene=scene)
        if not self._wait(self.apply_client.call_async(apply), 10).success:
            raise RuntimeError("MoveIt refused the workcell collision geometry")

    def _validate_trajectory(self, plan) -> dict:
        from moveit_msgs.srv import GetStateValidity
        trajectory = plan.planned_trajectory.joint_trajectory
        if (not trajectory.points or len(trajectory.joint_names) != len(self.joint_limits)
                or set(trajectory.joint_names) != set(self.joint_limits)):
            raise RuntimeError("Empty/incomplete arm trajectory")
        checks = 0
        previous, previous_time = None, -1.0
        for point in trajectory.points:
            values = np.asarray(point.positions, dtype=float)
            if values.shape != (len(trajectory.joint_names),) or not np.isfinite(values).all():
                raise RuntimeError("Invalid trajectory joint positions")
            stamp = stamp_seconds(point.time_from_start)
            if stamp < 0 or stamp <= previous_time:
                raise RuntimeError("Trajectory times must strictly increase")
            if point.velocities and len(point.velocities) != len(trajectory.joint_names):
                raise RuntimeError("Incomplete trajectory joint velocities")
            for index, name in enumerate(trajectory.joint_names):
                low, high, velocity = self.joint_limits[name]
                if not low-1e-5 <= values[index] <= high+1e-5:
                    raise RuntimeError(f"Joint limit violation: {name}")
                if point.velocities and (not math.isfinite(point.velocities[index]) or abs(point.velocities[index]) > velocity+1e-4):
                    raise RuntimeError(f"Joint velocity limit violation: {name}")
            steps = 1 if previous is None else max(1, math.ceil(float(np.max(np.abs(values-previous))) / .08))
            samples = [values] if previous is None else [previous+(values-previous)*fraction/steps for fraction in range(1, steps+1)]
            for sample in samples:
                request = GetStateValidity.Request()
                request.group_name = self.config.group_name
                request.robot_state = deepcopy(plan.trajectory_start)
                request.robot_state.joint_state.name = list(trajectory.joint_names)
                request.robot_state.joint_state.position = [float(v) for v in sample]
                request.robot_state.joint_state.velocity = []
                request.robot_state.joint_state.effort = []
                request.robot_state.is_diff = False
                response = self._wait(self.validity_client.call_async(request), 3.0)
                if not response.valid:
                    contacts = ",".join(f"{c.contact_body_1}/{c.contact_body_2}" for c in response.contacts[:3])
                    raise RuntimeError("Planned state is invalid/colliding: " + contacts)
                checks += 1
            previous, previous_time = values, stamp
        with self.lock:
            state = self.joint_state
        current = dict(zip(state.name, state.position))
        if any(abs(current[name]-trajectory.points[0].positions[i]) > .04 for i, name in enumerate(trajectory.joint_names)):
            raise RuntimeError("Robot moved after planning; trajectory start is stale")
        return {"joint_limits_checked": True, "collision_checked": True,
                "state_validity_samples": checks, "max_joint_sample_spacing_rad": .08,
                "continuous_collision_certification": False, "robot_motion_verified": False,
                "planned_joint_trajectory": {
                    "joint_names": list(trajectory.joint_names),
                    "points": [{"positions_rad": list(point.positions),
                                "velocities_rad_s": list(point.velocities),
                                "time_from_start_s": stamp_seconds(point.time_from_start)}
                               for point in trajectory.points]}}

    def _joint_goal(self, desired_tool: Pose):
        """Solve collision-aware IK using a reproducible numerical branch seed.

        An explicit joint goal avoids repeatedly sampling distant IK branches
        inside OMPL's pose-goal sampler. The resulting path still passes all
        trajectory and measured optical-pose checks.
        """
        from builtin_interfaces.msg import Duration
        from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
        from moveit_msgs.srv import GetPositionIK
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name, ik.ik_link_name = self.config.group_name, self.config.tool_frame
        ik.avoid_collisions = True
        ik.timeout = Duration(sec=1)
        ik.robot_state.joint_state.name = list(UR_ARM_JOINT_NAMES)
        ik.robot_state.joint_state.position = list(self.config.ik_seed_positions)
        ik.robot_state.is_diff = False
        ik.pose_stamped.header.frame_id = desired_tool.frame_id
        p, q = desired_tool.position, desired_tool.orientation
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = p
        ik.pose_stamped.pose.orientation.x, ik.pose_stamped.pose.orientation.y, ik.pose_stamped.pose.orientation.z, ik.pose_stamped.pose.orientation.w = q
        response = self._wait(self.ik_client.call_async(request), 4.)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"collision_aware_ik_error_{response.error_code.val}")
        solution = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        seeds = dict(zip(UR_ARM_JOINT_NAMES, self.config.ik_seed_positions))
        constraints = Constraints()
        for name, (low, high, _) in self.joint_limits.items():
            value = solution.get(name, float("nan"))
            if not math.isfinite(value):
                raise RuntimeError(f"IK joint bound violation: {name}")
            # Equivalent revolutions have identical FK. Keep the branch near
            # the declared seed instead of requesting an unnecessary full turn.
            equivalent = [value+turn*2*math.pi for turn in range(-2, 3) if low <= value+turn*2*math.pi <= high]
            if not equivalent:
                raise RuntimeError(f"IK joint bound violation: {name}")
            value = min(equivalent, key=lambda candidate: abs(candidate-seeds[name]))
            constraints.joint_constraints.append(JointConstraint(joint_name=name, position=value,
                                                                 tolerance_above=.001, tolerance_below=.001, weight=1.))
        return constraints

    def _plan_to(self, viewpoint: Viewpoint):
        """Prepare and validate a trajectory without executing any movement."""
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import MoveItErrorCodes
        started = time.monotonic()
        if time.monotonic()-self.clock_received > 3.0:
            return None, MotionResult(False, "verify", "simulation_clock_stalled")
        try:
            optical_in_tool = self._lookup(self.config.tool_frame, self.config.optical_frame)
            desired_tool = pose_from_matrix(pose_matrix(viewpoint.camera_pose) @ np.linalg.inv(pose_matrix(optical_in_tool)), self.config.world_frame)
            goal = MoveGroup.Goal()
            goal.request.group_name = self.config.group_name
            goal.request.pipeline_id = "ompl"
            goal.request.num_planning_attempts = 5
            goal.request.allowed_planning_time = self.config.planning_seconds
            goal.request.max_velocity_scaling_factor = self.config.velocity_scale
            goal.request.max_acceleration_scaling_factor = self.config.acceleration_scale
            goal.request.start_state.is_diff = True
            goal.request.goal_constraints = [self._joint_goal(desired_tool)]
            workspace = goal.request.workspace_parameters
            workspace.header.frame_id = self.config.world_frame
            workspace.min_corner.x, workspace.min_corner.y, workspace.min_corner.z = self.config.workspace_min
            workspace.max_corner.x, workspace.max_corner.y, workspace.max_corner.z = self.config.workspace_max
            goal.planning_options.plan_only = True
            goal.planning_options.replan = False
            handle = self._wait(self.plan_client.send_goal_async(goal), 10)
            if not handle.accepted:
                return None, MotionResult(False, "plan", "planning_goal_rejected")
            plan = self._wait(handle.get_result_async(), self.config.planning_seconds+15).result
            if plan.error_code.val != MoveItErrorCodes.SUCCESS:
                return None, MotionResult(False, "plan", f"planner_error_{plan.error_code.val}")
            checks = self._validate_trajectory(plan)
            points = plan.planned_trajectory.joint_trajectory.points
            positions = np.asarray([point.positions for point in points])
            path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            checks.update(planned_duration_s=stamp_seconds(points[-1].time_from_start),
                          planned_joint_path_length_rad=path, planning_wall_seconds=time.monotonic()-started)
            return plan, MotionResult(True, "plan", "planned_and_checked_without_execution", checks=checks)
        except Exception as exc:
            return None, MotionResult(False, "plan", f"plan_validation: {type(exc).__name__}: {exc}")

    def preview_viewpoint(self, viewpoint: Viewpoint) -> MotionResult:
        """Estimate checked motion cost; never send an execution action."""
        _, result = self._plan_to(viewpoint)
        return result

    def view_context(self) -> ViewContext:
        """Read measured camera state and actual MoveIt box-world geometry."""
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene
        from shape_msgs.msg import SolidPrimitive
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        world = self._wait(self.scene_client.call_async(request), 10).scene.world
        if world.octomap.octomap.data:
            raise RuntimeError("Visibility estimation currently requires a box-only world, without octomap")
        boxes = []
        def convert(value, frame, default_identity=False):
            q = (value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w)
            p = (value.position.x, value.position.y, value.position.z)
            # CollisionObject.pose may be the ROS default (all zero), which
            # MoveIt treats as the identity object origin.
            if default_identity and not any(q) and not any(p):
                q = (0., 0., 0., 1.)
            return Pose(p, q, frame)
        for obj in world.collision_objects:
            if obj.meshes or obj.planes or len(obj.primitives) != len(obj.primitive_poses):
                raise RuntimeError("Unsupported geometry in box-world visibility estimator")
            frame = obj.header.frame_id
            if not frame:
                raise RuntimeError("Visibility object has no reference frame")
            parent = np.eye(4) if frame == self.config.world_frame else pose_matrix(self._lookup(self.config.world_frame, frame))
            origin = parent @ pose_matrix(convert(obj.pose, frame, default_identity=True))
            for index, (primitive, relative) in enumerate(zip(obj.primitives, obj.primitive_poses)):
                if primitive.type != SolidPrimitive.BOX:
                    raise RuntimeError("Visibility estimation currently supports box world objects only")
                pose = pose_from_matrix(origin @ pose_matrix(convert(relative, frame)), self.config.world_frame)
                boxes.append(SceneBox(f"{obj.id}:{index}", pose, tuple(primitive.dimensions)))
        if not boxes:
            raise RuntimeError("No known world geometry available for active view selection")
        info = self.camera_info
        if any(abs(value) > 1e-10 for value in info.d):
            raise RuntimeError("Visibility projection requires the supplied distortion-free camera")
        return ViewContext(self._lookup(self.config.world_frame, self.config.optical_frame), tuple(info.k),
                           (info.width, info.height), tuple(boxes), "moveit_planning_scene_box_geometry", self.clock)

    def move_to(self, viewpoint: Viewpoint) -> MotionResult:
        """Replan from current state, execute, then verify measured camera pose."""
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import MoveItErrorCodes
        plan, prepared = self._plan_to(viewpoint)
        if not prepared.ok:
            return prepared
        checks = dict(prepared.checks)
        execute_handle = None
        try:
            execute_goal = ExecuteTrajectory.Goal(trajectory=plan.planned_trajectory)
            execute_handle = self._wait(self.execute_client.send_goal_async(execute_goal), 10)
            if not execute_handle.accepted:
                return MotionResult(False, "execute", "execution_goal_rejected", checks=checks)
            result = self._wait(execute_handle.get_result_async(), self.config.execution_timeout).result
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                return MotionResult(False, "execute", f"execution_error_{result.error_code.val}", checks=checks)
            settle_start = self.clock
            end = time.monotonic()+5.0
            while self.clock < settle_start+.3:
                if time.monotonic() > end:
                    raise TimeoutError("Simulation did not advance after execution")
                time.sleep(.02)
            actual = self._lookup(self.config.world_frame, self.config.optical_frame)
            distance, angle = pose_error(actual, viewpoint.camera_pose)
            checks.update(position_error_m=distance, orientation_error_rad=angle)
            if distance > self.config.position_tolerance or angle > self.config.orientation_tolerance:
                return MotionResult(False, "verify", "executed_camera_pose_outside_tolerance", checks=checks)
            checks["robot_motion_verified"] = True
            return MotionResult(True, "verify", "planned_checked_executed_and_measured", self.clock, actual, checks)
        except Exception as exc:
            if execute_handle is not None and execute_handle.accepted:
                try:
                    self._wait(execute_handle.cancel_goal_async(), 5)
                except Exception:
                    pass
            return MotionResult(False, "execute", f"execution_failure: {type(exc).__name__}: {exc}", checks=checks)

    def capture(self, after_stamp: float, timeout: float) -> Observation:
        end = time.monotonic()+timeout
        last_reason = "camera_timeout"
        while self.rclpy.ok() and time.monotonic() < end:
            with self.lock:
                message = self.image_message
                info = self.camera_info
            if message is None or stamp_seconds(message.header.stamp) <= after_stamp:
                time.sleep(.02)
                continue
            if message.header.frame_id != self.config.optical_frame:
                raise AcquisitionError(f"image_frame_mismatch:{message.header.frame_id}")
            if info is None or (info.width, info.height) != (message.width, message.height):
                raise AcquisitionError("camera_calibration_mismatch")
            try:
                pose = self._lookup(self.config.world_frame, self.config.optical_frame, message.header.stamp)
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8").copy()
                return Observation(image, stamp_seconds(message.header.stamp), message.header.frame_id, pose)
            except Exception as exc:
                last_reason = f"image_or_tf_unavailable:{type(exc).__name__}:{exc}"
                time.sleep(.02)
        raise AcquisitionError(last_reason)

    def close(self) -> None:
        self.rclpy.try_shutdown()
        self.executor.shutdown(timeout_sec=3.0)
        self.thread.join(timeout=3.0)
        self.listener.unregister()
        self.plan_client.destroy()
        self.execute_client.destroy()
        self.controller_client.destroy()
        self.node.destroy_node()


def run_robot(config: Config, output: Path, scenario: str, ocr_model: Path | None = None, ocr_backend: str = "svm") -> int:
    """Run one real Gazebo inspection and record startup failures as evidence."""
    import rclpy
    rclpy.init(args=[])
    backend = None
    started = time.monotonic()
    metadata = {"backend": "gazebo_moveit", "scenario": scenario, "config": config.to_dict(),
                "ground_truth_source": "generated_workcell_metadata", "simulation_only": True}
    try:
        with RunRecorder(output, metadata) as recorder:
            try:
                backend = RosBackend(config, scenario)
                recorder.write_json("simulation.json", backend.metadata)
                with (recorder.directory / "robot_description.urdf").open("x") as description:
                    description.write(backend.description)
                info = backend.camera_info
                recorder.write_json("camera_info.json", {"frame_id": info.header.frame_id, "width": info.width,
                                    "height": info.height, "k": list(info.k), "d": list(info.d),
                                    "distortion_model": info.distortion_model})
                if config.marking_kind == "text":
                    from .ocr import LineRecognizer
                    from .text_inspection import TextInspector
                    if any(abs(value) > 1e-10 for value in info.d):
                        raise ValueError("Text ROI requires the supplied distortion-free camera")
                    recognizer = LineRecognizer.load(ocr_model) if ocr_backend == "svm" else LineRecognizer()
                    recorder.write_json("ocr_model.json", recognizer.metadata)
                    inspector = TextInspector(config, recognizer, list(info.k))
                else:
                    inspector = MarkerInspector(config)
                result = InspectionRunner(config, backend, inspector, recorder).run()
            except Exception as exc:
                result = InspectionResult(Outcome.ERROR, f"startup: {type(exc).__name__}: {exc}",
                                          0, 0, time.monotonic()-started, "gazebo_moveit")
                recorder.event("STARTUP_ERROR", result=result)
                recorder.write_json("result.json", result.to_dict())
            render_report(output)
            print(json.dumps({**result.to_dict(), "run_directory": str(output)}, indent=2))
            return 0 if result.outcome in (Outcome.ACCEPT, Outcome.REJECT) else (3 if result.outcome == Outcome.ESCALATE else 2)
    finally:
        if backend is not None:
            backend.close()
        rclpy.try_shutdown()
