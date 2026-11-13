"""Collect all configured views through the existing simulation safety gates."""

import json
import math
from pathlib import Path

from .config import Config
from .dataset import file_sha256
from .geometry import pose_error
from .markings import project_text_roi, rectify_text_roi
from .models import AcquisitionError
from .policy import validate_viewpoint
from .recording import RunRecorder, json_value


def capture_case(case: dict, output: Path) -> int:
    """Save actual camera views; make no OCR or accept/reject decision.

    The parent process owns the simulator lifecycle. This function connects
    exclusively through RosBackend's simulation-only checks.
    """
    import rclpy
    from .ros_backend import RosBackend
    config = Config.from_dict(case["config"])
    rclpy.init(args=[])
    backend = None
    samples, failures = [], []
    metadata = {"task": "text_dataset_capture", "backend": "gazebo_moveit", "case": case,
                "scenario": case["scenario"], "config": config.to_dict(), "simulation_only": True}
    try:
        with RunRecorder(output, metadata) as recorder:
            recorder.event("START", task="text_dataset_capture")
            try:
                backend = RosBackend(config, case["scenario"])
                recorder.write_json("simulation.json", backend.metadata)
                with (output / "robot_description.urdf").open("x") as stream:
                    stream.write(backend.description)
                info = backend.camera_info
                calibration = {"frame_id": info.header.frame_id, "width": info.width, "height": info.height,
                               "k": list(info.k), "d": list(info.d), "distortion_model": info.distortion_model}
                if any(abs(v) > 1e-10 for v in calibration["d"]):
                    raise ValueError("Known-fixture rectification currently requires the supplied distortion-free camera")
                recorder.write_json("camera_info.json", calibration)
                for index, viewpoint in enumerate(config.viewpoints()[:config.max_views], 1):
                    validate_viewpoint(viewpoint, config)
                    print(f"{case['case_id']}: moving to {viewpoint.name}", flush=True)
                    recorder.event("MOTION_REQUEST", viewpoint=viewpoint)
                    motion = backend.move_to(viewpoint)
                    recorder.event("MOTION_RESULT", viewpoint=viewpoint.name, motion=motion)
                    if not motion.ok:
                        failures.append({"view": viewpoint.name, "reason": motion.reason, "phase": motion.phase})
                        if motion.phase != "plan":
                            break
                        continue
                    try:
                        if motion.actual_camera_pose is None or not math.isfinite(motion.arrived_at):
                            raise RuntimeError("Missing measured motion evidence")
                        observation = backend.capture(motion.arrived_at, config.image_timeout)
                        if observation.frame_id != config.optical_frame or not math.isfinite(observation.timestamp) or observation.timestamp <= motion.arrived_at:
                            raise AcquisitionError("Stale or wrongly framed image")
                        distance, angle = pose_error(observation.camera_pose, viewpoint.camera_pose)
                        if distance > config.position_tolerance or angle > config.orientation_tolerance:
                            raise RuntimeError("Image pose drifted after motion")
                        name = recorder.image(observation.image, f"view_{index:02d}.png")
                        corners = project_text_roi(config, observation.camera_pose, calibration["k"])
                        crop = rectify_text_roi(observation.image, corners)
                        crop_name = recorder.image(crop, f"view_{index:02d}_rectified.png")
                        row = {key: case[key] for key in ("case_id", "part_id", "split", "scenario", "seed", "expected_text", "surface_text")}
                        row.update(sample_id=f"{case['case_id']}_view{index:02d}", viewpoint=viewpoint.name,
                                   source="gazebo_robot_camera", simulation_only=True, image=name, rectified_image=crop_name,
                                   image_sha256=file_sha256(output / name), rectified_sha256=file_sha256(output / crop_name),
                                   image_stamp=observation.timestamp, motion_arrived_at=motion.arrived_at,
                                   optical_frame=observation.frame_id, camera_pose=observation.camera_pose.to_dict(),
                                   camera_info=calibration, text_polygon_px=corners.tolist(),
                                   roi_source="known_fixture_geometry_and_measured_camera_pose", readability_label=None,
                                   readability_label_source="unreviewed", label_matches_expected=case["expected_text"] == case["surface_text"],
                                   motion_checks={k: v for k, v in motion.checks.items() if k != "planned_joint_trajectory"})
                        samples.append(json_value(row))
                        recorder.event("CAPTURE", sample=row)
                        print(f"{case['case_id']}: saved {name}; readability unreviewed", flush=True)
                    except (AcquisitionError, ValueError) as exc:
                        failures.append({"view": viewpoint.name, "reason": str(exc), "phase": "acquire_or_crop"})
                        recorder.event("CAPTURE_FAILURE", viewpoint=viewpoint.name, reason=str(exc))
                    except Exception as exc:
                        failures.append({"view": viewpoint.name, "reason": str(exc), "phase": "verify"})
                        recorder.event("CAPTURE_FAILURE", viewpoint=viewpoint.name, reason=str(exc))
                        break
            except Exception as exc:
                failures.append({"phase": "startup_or_capture", "reason": f"{type(exc).__name__}:{exc}"})
                recorder.event("EXCEPTION", reason=str(exc))
            complete = len(samples) == config.max_views and not failures
            result = {"case_id": case["case_id"], "status": "COMPLETE" if complete else "INCOMPLETE",
                      "images": len(samples), "failures": failures, "inspection_decision": None}
            recorder.write_json("samples.json", samples)
            recorder.write_json("capture_result.json", result)
            recorder.event("FINISH", result=result)
            print(json.dumps(result), flush=True)
            return 0 if complete else 2
    finally:
        if backend is not None:
            backend.close()
        rclpy.try_shutdown()
