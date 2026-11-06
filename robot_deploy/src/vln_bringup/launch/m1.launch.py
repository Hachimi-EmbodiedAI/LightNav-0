"""Launch the complete VLN stack on the Genisom M1 (铜锤).

DIFFERENCES FROM ``tron.launch.py`` / ``go2.launch.py``

  1. NO CAMERA LAUNCH.  The M1's camera is published by the vendor stack
     (``/camera/front/image_raw`` and ``.../compressed``).  Start it separately;
     this file does not own it.
  2. ``robot_control`` IS A PREREQUISITE, not something we launch.  It is the
     vendor's prebuilt node (no source in the repo) and it must already be
     running: it provides ``/cmd_vel``, ``/robot_control/odometry``,
     ``/robot_control/state`` and the posture services.
  3. IMAGE TRANSPORT IS ``compressed``.  Two independent reasons:
       - the camera publishes ``bgr8`` and ``vln_client``'s raw path only
         accepts ``rgb8`` (it drops anything else with a warning);
       - ``raw`` pulls in ``cv_bridge``, which breaks under numpy 2.x.
       ``compressed`` goes through ``cv2.imdecode`` and avoids both.
  4. SPEEDS ARE MUCH LOWER.  Inference runs at ~1.5 Hz, so the MPC sees a new
     waypoint every ~650 ms.  The TRON defaults (1.5 m/s) were tuned for a
     faster loop and higher-performing hardware.  Measured on this M1
     (hachimi_ws): commanding 0.3 m/s yields ~0.22 m/s of real travel, about
     73% -- the rest is gait slip.

MEASURED BASELINE (AGX Orin 64 GB, MODE_50W, vLLM local):
    ~659 ms/step end to end = 1.52 Hz, ViT 127 ms + LLM decode 526 ms.
    At 0.3 m/s the robot moves ~0.20 m between consecutive inferences.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    server_url = LaunchConfiguration("server_url")
    image_topic = LaunchConfiguration("image_topic")
    image_transport = LaunchConfiguration("image_transport")
    track_v_max = LaunchConfiguration("track_v_max")
    objnav_v_max = LaunchConfiguration("objnav_v_max")
    max_linear = LaunchConfiguration("max_linear")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "server_url",
                default_value="ws://127.0.0.1:8050",
                description="LightNav inference WebSocket server",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera/front/image_raw",
                description=(
                    "Base topic; do NOT append /compressed -- image_transport "
                    "does that when transport=compressed"
                ),
            ),
            DeclareLaunchArgument(
                "image_transport",
                default_value="compressed",
                description="compressed (JPEG, numpy-2.x safe) or raw (needs rgb8)",
            ),
            DeclareLaunchArgument(
                "track_v_max",
                default_value="0.3",
                description="Tracking-mode speed cap, m/s (~1.5 Hz inference)",
            ),
            DeclareLaunchArgument(
                "objnav_v_max",
                default_value="0.25",
                description="Object-goal speed cap, m/s",
            ),
            DeclareLaunchArgument(
                "max_linear",
                default_value="0.5",
                description="Hard clamp applied inside m1_adapter, m/s; 0 disables",
            ),
            # --- perception -> inference ---------------------------------
            Node(
                package="vln_client",
                executable="vln_client",
                name="vln_client",
                output="screen",
                parameters=[
                    {
                        "server_url": server_url,
                        "image_topic": image_topic,
                        "image_transport": image_transport,
                    }
                ],
            ),
            # --- operator UI ---------------------------------------------
            Node(
                package="vln_web",
                executable="vln_web",
                name="vln_web",
                output="screen",
                parameters=[
                    {
                        # Manual (WASD) limits are deliberately tighter than on
                        # TRON: this is a first bring-up on a real robot.
                        "manual_linear_limit": 0.5,
                        "manual_angular_limit": 1.0,
                        "manual_linear_accel": 0.5,
                        "manual_angular_accel": 2.0,
                    }
                ],
            ),
            # --- control --------------------------------------------------
            Node(
                package="vln_mpc",
                executable="vln_mpc",
                name="vln_mpc",
                output="screen",
                parameters=[
                    {
                        "track_v_max": track_v_max,
                        "objnav_v_max": objnav_v_max,
                        "w_max": 1.0,
                        "a_max_v": 1.0,
                        "a_max_w": 2.0,
                        "q_x": 10.0,
                        "q_y": 10.0,
                        "q_yaw": 1.0,
                        "r_v": 0.1,
                        "r_w": 0.1,
                        "v_output_scale": 1.0,
                        "w_output_scale": 1.0,
                        # robot_control owns odometry; remap instead of
                        # re-publishing it in the adapter.
                        "odom_topic": "/robot_control/odometry",
                    }
                ],
            ),
            # --- robot -----------------------------------------------------------------
            Node(
                package="m1_adapter",
                executable="m1_adapter",
                name="m1_adapter",
                output="screen",
                parameters=[
                    {
                        "state_topic": "/robot_control/state",
                        "cmd_vel_topic": "/cmd_vel",
                        "control_rate_hz": 20.0,
                        # < robot_control's command_timeout_ms (300 ms) so our
                        # own watchdog never outlives the vendor one.
                        "input_watchdog_s": 0.25,
                        "state_timeout_s": 1.0,
                        "max_linear": max_linear,
                        "max_angular": 1.0,
                        "auto_take_control": True,
                        "take_control_retry_s": 5.0,
                    }
                ],
            ),
        ]
    )
