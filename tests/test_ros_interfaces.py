"""Check actual installed ROS message interfaces without commanding any robot."""

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy", reason="Source /opt/ros/jazzy/setup.bash for ROS interface tests")
from builtin_interfaces.msg import Duration
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotState, RobotTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from activevisionqa.config import Config
from activevisionqa.models import Pose
from activevisionqa.ros_backend import RosBackend, UR_ARM_JOINT_NAMES


def test_ik_preserves_tool_pose_frame_and_requires_collision_aware_solution():
    backend, _ = _validator()
    requests = []
    def respond(request):
        requests.append(request)
        return SimpleNamespace(error_code=SimpleNamespace(val=1), solution=RobotState(
            joint_state=JointState(name=list(backend.joint_limits), position=[.2]*6)))
    backend.ik_client = SimpleNamespace(call_async=respond)
    goal = backend._joint_goal(Pose((.5, .1, .6), (0., 0., 0., 1.)))
    ik = requests[0].ik_request
    assert ik.avoid_collisions and ik.ik_link_name == "tool0"
    assert ik.pose_stamped.header.frame_id == "world"
    assert ik.pose_stamped.pose.position.z == .6 and ik.pose_stamped.pose.orientation.w == 1.
    assert list(ik.robot_state.joint_state.position) == list(Config().ik_seed_positions)
    assert len(goal.joint_constraints) == 6
    assert all(joint.position == .2 for joint in goal.joint_constraints)


@pytest.mark.parametrize("fault", ["failure", "missing_joint", "bound"])
def test_failed_or_incomplete_ik_cannot_become_a_motion_goal(fault):
    backend, _ = _validator()
    positions = [2. if fault == "bound" else .2]*6
    names = list(backend.joint_limits)[:5 if fault == "missing_joint" else 6]
    response = SimpleNamespace(error_code=SimpleNamespace(val=-31 if fault == "failure" else 1),
                               solution=RobotState(joint_state=JointState(name=names, position=positions[:len(names)])))
    backend.ik_client = SimpleNamespace(call_async=lambda request: response)
    with pytest.raises(RuntimeError, match="ik_error|bound violation"):
        backend._joint_goal(Pose((.5, .1, .6), (0., 0., 0., 1.)))


def test_equivalent_joint_revolution_uses_declared_ik_branch():
    import math
    backend, _ = _validator()
    backend.joint_limits["wrist_3_joint"] = (-6.1, 6.1, 3.)
    positions = [.2]*5 + [3.446]
    response = SimpleNamespace(error_code=SimpleNamespace(val=1), solution=RobotState(
        joint_state=JointState(name=list(UR_ARM_JOINT_NAMES), position=positions)))
    backend.ik_client = SimpleNamespace(call_async=lambda request: response)
    goal = backend._joint_goal(Pose((.5, .1, .6), (0., 0., 0., 1.)))
    assert goal.joint_constraints[-1].position == pytest.approx(3.446-2*math.pi)


def test_execute_goal_uses_the_jazzy_interface():
    goal = ExecuteTrajectory.Goal(trajectory=RobotTrajectory())
    assert "trajectory" in goal.get_fields_and_field_types()


def _validator():
    import threading
    backend = object.__new__(RosBackend)
    backend.config = Config()
    backend.joint_limits = {name: (-1., 1., 1.) for name in UR_ARM_JOINT_NAMES}
    backend.lock = threading.Lock()
    names = list(backend.joint_limits)
    backend.joint_state = JointState(name=names, position=[0.]*6)
    backend._wait = lambda future, timeout: future
    backend.validity_client = SimpleNamespace(call_async=lambda request: SimpleNamespace(valid=True))
    trajectory = RobotTrajectory(joint_trajectory=JointTrajectory(joint_names=names, points=[
        JointTrajectoryPoint(positions=[0.]*6, time_from_start=Duration(sec=0)),
        JointTrajectoryPoint(positions=[.2]*6, velocities=[.1]*6, time_from_start=Duration(sec=2)),
    ]))
    plan = SimpleNamespace(planned_trajectory=trajectory, trajectory_start=RobotState())
    return backend, plan


def test_trajectory_limits_and_collision_services_gate_execution():
    backend, plan = _validator()
    checks = backend._validate_trajectory(plan)
    assert checks["state_validity_samples"] >= 4
    assert checks["collision_checked"] and checks["joint_limits_checked"]
    assert len(checks["planned_joint_trajectory"]["points"]) == 2
    plan.planned_trajectory.joint_trajectory.points[1].positions[0] = 2.
    with pytest.raises(RuntimeError, match="Joint limit"):
        backend._validate_trajectory(plan)


def test_colliding_state_or_stale_start_is_rejected():
    backend, plan = _validator()
    backend.validity_client = SimpleNamespace(call_async=lambda request: SimpleNamespace(valid=False, contacts=[]))
    with pytest.raises(RuntimeError, match="colliding"):
        backend._validate_trajectory(plan)
    backend, plan = _validator()
    backend.joint_state.position[0] = .5
    with pytest.raises(RuntimeError, match="stale"):
        backend._validate_trajectory(plan)


@pytest.mark.parametrize("fault,reason", [("duplicate", "incomplete"), ("negative_time", "times"),
                                          ("velocities", "velocities")])
def test_malformed_trajectory_is_rejected_before_execution(fault, reason):
    backend, plan = _validator()
    trajectory = plan.planned_trajectory.joint_trajectory
    if fault == "duplicate":
        trajectory.joint_names.append(trajectory.joint_names[0])
    if fault == "negative_time":
        trajectory.points[0].time_from_start.sec = -1
    if fault == "velocities":
        trajectory.points[1].velocities = [.1]
    with pytest.raises(RuntimeError, match=reason):
        backend._validate_trajectory(plan)


def test_previews_are_plan_only_and_do_not_grant_stale_execution_permission():
    import time
    from moveit_msgs.msg import Constraints, MoveItErrorCodes
    backend, plan = _validator()
    plan.error_code = MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS)
    backend.clock_received = time.monotonic()
    backend._lookup = lambda *args: Pose((0., 0., .08), (0., 0., 0., 1.), "tool0")
    backend._joint_goal = lambda pose: Constraints()
    goals = []
    def send(goal):
        goals.append(goal)
        return SimpleNamespace(accepted=True, get_result_async=lambda: SimpleNamespace(result=plan))
    backend.plan_client = SimpleNamespace(send_goal_async=send)
    backend.execute_client = SimpleNamespace(send_goal_async=lambda goal: pytest.fail("Unchecked or stale preview was executed"))
    preview = backend.preview_viewpoint(backend.config.viewpoints()[0])
    assert preview.ok and preview.phase == "plan"
    assert goals[0].planning_options.plan_only
    assert preview.checks["planned_duration_s"] == 2.
    assert preview.checks["planned_joint_path_length_rad"] == pytest.approx((6*.2**2)**.5)
    backend.joint_state.position[0] = .5
    result = backend.move_to(backend.config.viewpoints()[0])
    assert len(goals) == 2 and not result.ok and "stale" in result.reason


@pytest.mark.parametrize("unsupported", [False, True])
def test_view_context_composes_object_and_primitive_frames_or_rejects_unknown_geometry(unsupported):
    import math
    from geometry_msgs.msg import Pose as RosPose
    from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
    from shape_msgs.msg import SolidPrimitive
    backend = object.__new__(RosBackend)
    backend.config, backend.clock = Config(), 12.
    backend.camera_info = SimpleNamespace(k=[692., 0., 400., 0., 692., 300., 0., 0., 1.], width=800, height=600, d=[])
    backend._wait = lambda future, timeout: future
    backend._lookup = lambda *args: backend.config.viewpoints()[0].camera_pose
    obj = CollisionObject()
    obj.header.frame_id, obj.id = "world", "unlabeled_box"
    obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = .3, .4, .1
    obj.pose.orientation.z = obj.pose.orientation.w = math.sqrt(.5)
    relative = RosPose()
    relative.position.x, relative.position.z, relative.orientation.w = .1, .2, 1.
    primitive = SolidPrimitive(type=SolidPrimitive.SPHERE if unsupported else SolidPrimitive.BOX,
                               dimensions=[.1] if unsupported else [.2, .2, .4])
    obj.primitives, obj.primitive_poses = [primitive], [relative]
    response = SimpleNamespace(scene=PlanningScene(world=PlanningSceneWorld(collision_objects=[obj])))
    backend.scene_client = SimpleNamespace(call_async=lambda request: response)
    if unsupported:
        with pytest.raises(RuntimeError, match="box world objects"):
            backend.view_context()
    else:
        context = backend.view_context()
        assert context.simulation_time == 12.
        assert context.boxes[0].pose.position == pytest.approx((.3, .5, .3))
        assert context.boxes[0].pose.frame_id == "world"
