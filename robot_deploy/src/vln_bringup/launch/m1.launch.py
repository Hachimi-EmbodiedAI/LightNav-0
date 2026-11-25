"""Launch the complete VLN stack on the Genisom M1 (铜锤), standalone mode.

DIFFERENCES FROM ``tron.launch.py`` / ``go2.launch.py``

  1. ``m1_robot_driver`` (SDK 0.2.3 in ``hachimi_ws``) IS A PREREQUISITE, not
     launched here.  Start it separately with::

         source /home/jetson/Public/hachimi_ws/scripts/m1_driver_env.sh
         ros2 launch m1_robot_driver robot_control.launch.py

     It provides ``/cmd_vel``, ``/robot_control/odometry``,
     ``/robot_control/state`` and the posture/mode services.
  2. OPTIONAL STANDALONE RTSP CAMERA (``start_rtsp_camera:=true`` by default).
     The M1 front camera is exposed over RTSP
     (``rtsp://192.168.168.168:8554/front``).  ``m1_rtsp_camera`` pulls it and
     publishes JPEG ``CompressedImage`` on ``/camera/front/image_raw/compressed``
     so this stack runs independently of ``zsm_console``.  Set
     ``start_rtsp_camera:=false`` if another node already publishes that topic.
  3. IMAGE TRANSPORT IS ``compressed`` AND ``image_topic`` MUST INCLUDE
     ``/compressed``.  Both ``vln_client`` and ``vln_web`` subscribe directly to
     ``image_topic`` with ``sensor_msgs/CompressedImage`` (they do not append
     ``/compressed`` automatically).  Using ``compressed`` avoids the ``bgr8``
     vs ``rgb8`` mismatch and keeps ``cv_bridge`` out of the hot path.
  4. SPEEDS ARE TUNED FOR ~1.0–1.5 HZ ONBOARD INFERENCE.  On AGX Orin 64 GB
     (MODE_50W, vLLM local), inference runs at ~650–950 ms/step, so angular
     velocity is capped conservatively (``w_max=0.4 rad/s``, ``r_w=0.5``) to
     avoid left-right overshoot, while linear caps default to ``0.5 m/s``
     (tracking) / ``0.4 m/s`` (objnav).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    server_url = LaunchConfiguration("server_url")
    image_topic = LaunchConfiguration("image_topic")
    image_transport = LaunchConfiguration("image_transport")
    start_rtsp_camera = LaunchConfiguration("start_rtsp_camera")
    rtsp_url = LaunchConfiguration("rtsp_url")
    web_port = LaunchConfiguration("web_port")
    track_v_max = LaunchConfiguration("track_v_max")
    objnav_v_max = LaunchConfiguration("objnav_v_max")
    w_max = LaunchConfiguration("w_max")
    max_linear = LaunchConfiguration("max_linear")
    max_angular = LaunchConfiguration("max_angular")
    auto_take_control = LaunchConfiguration("auto_take_control")
    posture_timeout_s = LaunchConfiguration("posture_timeout_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "server_url",
                default_value="ws://127.0.0.1:8050",
                description="LightNav inference WebSocket server",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera/front/image_raw/compressed",
                description=(
                    "Exact image topic subscribed by vln_client and vln_web; "
                    "include /compressed when image_transport=compressed"
                ),
            ),
            DeclareLaunchArgument(
                "image_transport",
                default_value="compressed",
                description="compressed (JPEG, recommended) or raw (needs rgb8)",
            ),
            DeclareLaunchArgument(
                "start_rtsp_camera",
                default_value="true",
                description=(
                    "Launch standalone m1_rtsp_camera to publish RTSP front "
                    "camera as CompressedImage (independent of zsm_console)"
                ),
            ),
            DeclareLaunchArgument(
                "rtsp_url",
                default_value="rtsp://192.168.168.168:8554/front",
                description="M1 front camera RTSP URL for m1_rtsp_camera",
            ),
            DeclareLaunchArgument(
                "web_port",
                default_value="8088",
                description="HTTP port for vln_web (kept separate from 8080/8099)",
            ),
            DeclareLaunchArgument(
                "track_v_max",
                default_value="0.5",
                description="Tracking-mode speed cap, m/s",
            ),
            DeclareLaunchArgument(
                "objnav_v_max",
                default_value="0.4",
                description="Object-goal speed cap, m/s",
            ),
            DeclareLaunchArgument(
                "w_max",
                default_value="0.4",
                description=(
                    "MPC angular velocity cap, rad/s (~23 deg/s to avoid "
                    "left-right overshoot under ~1 Hz onboard inference)"
                ),
            ),
            DeclareLaunchArgument(
                "max_linear",
                default_value="0.7",
                description="Hard linear clamp inside m1_adapter, m/s; 0 disables",
            ),
            DeclareLaunchArgument(
                "max_angular",
                default_value="0.6",
                description="Hard angular clamp inside m1_adapter, rad/s; 0 disables",
            ),
            DeclareLaunchArgument(
                "auto_take_control",
                default_value="true",
                description="Call /robot_control/take_control on startup",
            ),
            DeclareLaunchArgument(
                "posture_timeout_s",
                default_value="10.0",
                description="Timeout waiting for m1_robot_driver posture/mode services",
            ),
            # --- optional standalone RTSP front camera --------------------
            Node(
                package="m1_adapter",
                executable="m1_rtsp_camera",
                name="m1_rtsp_camera",
                output="screen",
                condition=IfCondition(start_rtsp_camera),
                parameters=[
                    {
                        "rtsp_url": rtsp_url,
                        "image_topic": image_topic,
                        "frame_id": "front_camera_optical",
                        "fps": 10.0,
                        "output_width": 640,
                        "jpeg_quality": 80,
                    }
                ],
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
                        "port": ParameterValue(web_port, value_type=int),
                        "image_topic": image_topic,
                        "image_transport": image_transport,
                        "odom_topic": "/robot_control/odometry",
                        # m1_adapter publishes TwistStamped on cmd_vel_sent
                        # (/cmd_vel is plain Twist for m1_robot_driver).
                        "sent_command_topic": "cmd_vel_sent",
                        # Manual (WASD) limits are deliberately tighter than on
                        # TRON: this is a first bring-up on a real robot.
                        "manual_linear_limit": 0.6,
                        "manual_angular_limit": 0.6,
                        "manual_linear_accel": 0.6,
                        "manual_angular_accel": 1.0,
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
                        "track_v_max": ParameterValue(track_v_max, value_type=float),
                        "objnav_v_max": ParameterValue(objnav_v_max, value_type=float),
                        "w_max": ParameterValue(w_max, value_type=float),
                        "a_max_v": 1.0,
                        "a_max_w": 0.8,
                        "q_x": 10.0,
                        "q_y": 10.0,
                        "q_yaw": 1.0,
                        "r_v": 0.1,
                        "r_w": 0.5,
                        "v_output_scale": 1.0,
                        "w_output_scale": 1.0,
                        # m1_robot_driver owns odometry; remap directly.
                        "odom_topic": "/robot_control/odometry",
                    }
                ],
            ),
            # --- robot ----------------------------------------------------
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
                        # < m1_robot_driver's command_timeout_ms (300 ms).
                        "input_watchdog_s": 0.25,
                        "state_timeout_s": 1.0,
                        "posture_timeout_s": ParameterValue(
                            posture_timeout_s, value_type=float
                        ),
                        "max_linear": ParameterValue(max_linear, value_type=float),
                        "max_angular": ParameterValue(max_angular, value_type=float),
                        "auto_take_control": ParameterValue(
                            auto_take_control, value_type=bool
                        ),
                        "take_control_retry_s": 5.0,
                    }
                ],
            ),
        ]
    )
