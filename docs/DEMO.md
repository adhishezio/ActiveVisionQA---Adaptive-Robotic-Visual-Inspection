# Two-minute prototype demonstration

This walkthrough runs the original two-view marker baseline. For adaptive text inspection, use [ACTIVE_VISION.md](ACTIVE_VISION.md). Build once with `bash scripts/build.sh` and use Ubuntu terminals outside a Python virtual environment.

1. Start the workcell in terminal A:

   ```bash
   bash scripts/run_simulation.sh scenario:=occluded
   ```

   Wait for the simulated UR5e and camera to load. The inspector also waits for readiness, up to its bounded timeout.

   This command loads the workcell. The arm waits until the separate inspection command in terminal B starts a task. For a closer view, right-click the robot in the 3D scene, choose **Move to**, and use the mouse wheel to zoom. These camera controls change your GUI view, not the inspection camera's mounting or robot pose.

2. In terminal B, show the fixed-view baseline:

   ```bash
   bash scripts/inspect_robot.sh --scenario occluded --max-views 1
   ```

   Explain: “The camera cannot read the identifier from this blocked view. With one view allowed, the inspector escalates instead of guessing.” `ESCALATE` intentionally exits with code 3.

3. Allow the second viewpoint:

   ```bash
   bash scripts/inspect_robot.sh --scenario occluded --max-views 2
   ```

   This runs one bounded inspection and then exits; the arm stays at its final pose afterward. It is not a continuous animation. Open the printed output directory's `report.html` to see the actual inspection-camera images; the main Gazebo viewport shows the external scene camera.

   Explain: “Insufficient visual evidence triggers another planned camera pose. MoveIt validates a trajectory; the application checks its joint limits, sampled collision states and measured end pose before inspecting a fresh image.”

4. Open `report.html` inside the output directory printed by the second command. Show the blocked first image, readable second image and `ACCEPT` result. `events.jsonl` holds the trajectory and timestamp/pose evidence behind the report.

5. State the boundary: “This is a simulation test of a two-view marker inspection workflow. Real markings, factory conditions and hardware still require validation.” Stop terminal A with Ctrl+C.

For a broader evidence check, run `bash scripts/validate_simulation.sh --include-unreachable`. For every run, inspect the actual result; expected outcomes are not substituted when planning/rendering fails. The recorded successful examples and limitations are in [VALIDATION.md](VALIDATION.md).
