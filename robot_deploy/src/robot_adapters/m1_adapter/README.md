# m1_adapter

Bridges the generic LightNav robot interface (`TwistStamped` + posture services)
to the Genisom M1 / 铜锤 vendor stack (`/cmd_vel` as plain `Twist` +
`/robot_control/*`).

```text
vln_mpc ──mpc/cmd_vel (TwistStamped)──┐
vln_web ──web/cmd_vel (TwistStamped)──┤
                                      ├─→ m1_adapter ──/cmd_vel (Twist)──→ robot_control ──→ SDK
/robot_control/state (RobotStatus) ───┘        │
                                               └── gates: posture · e-stop · watchdog
```

## Why it is not a one-line type conversion

Three constraints the generic stack does not know about. Each is a real source
of "the robot silently does nothing" if ignored.

### 1. Posture gate

`genisom_robot_sdk/docs/zh/sdk_state_zh.md` documents which postures accept
`Move()`. Everything else either ignores a velocity command or — worse —
silently transitions the robot.

| Can move | Cannot move |
|---|---|
| `walk` `stand_up` `stair` `crawl_walk` `climb` `gait` `slim` `sk_walk` `sand` | `lie_down` `crawl` `balance_stand` `locked` |

`robot_control` publishes these as **lowercase snake_case** in
`RobotStatus.motion_status_name`; the SDK enum is UPPERCASE. `normalise_posture()`
accepts either, plus the CamelCase enum spelling that loses its underscore
(`CrawlWalk`).

Also: `StandUp()` reports **`WALK`** on completion, not `STAND_UP` — `STAND_UP`
only appears while the transition is in flight. So "is the robot standing?" must
accept `walk`.

### 2. Soft e-stop gate

`/robot_control/soft_emergency_stop` is **`std_srvs/SetBool`**, while every other
`robot_control` service is `Trigger`. Declaring it as `Trigger` makes the e-stop
path silently dead code. (`hachimi_ws` hit exactly this and documents it as a
BUGFIX; that is why the type is called out here and covered by a test.)

While engaged, every velocity command is forced to zero regardless of source.

### 3. Control ownership

The SDK grants control to one client at a time, and **an attached phone APP
always wins** — `TakeControl()` cannot preempt it. The adapter calls
`/robot_control/take_control` on startup and retries every 5 s, logging loudly
when refused. Without this, commands go nowhere and nothing complains.

## Interface

Subscribes:

| Topic | Type | Notes |
|---|---|---|
| `web/cmd_vel` | `TwistStamped` | manual (web UI) |
| `mpc/cmd_vel` | `TwistStamped` | automatic (MPC) |
| `/robot_control/state` | `robot_control/RobotStatus` | posture, e-stop, control source |

Publishes:

| Topic | Type | Notes |
|---|---|---|
| `/cmd_vel` | **`geometry_msgs/Twist`** | what `robot_control` actually subscribes to |
| `cmd_vel_sent` | `TwistStamped` | what was forwarded (for debugging) |
| `control/source` | `String` | `disabled` \| `manual` \| `auto` (latched) |
| `robot/events` | `String` | posture / e-stop / source transitions |
| `diagnostics` | `DiagnosticArray` | gate state, state age, clamp values |

Services:

| Service | Type | Maps to |
|---|---|---|
| `control/set_manual` | `SetBool` | local source switch |
| `control/set_auto` | `SetBool` | local source switch |
| `control/stop` | `Trigger` | disable + zero velocity |
| `robot/stand` | `Trigger` | `/robot_control/stand_up` |
| `robot/walk` | `Trigger` | `/robot_control/stand_up` (same service) |
| `robot/sit` | `Trigger` | `/robot_control/lie_down` |
| `robot/crawl` | `Trigger` | `/robot_control/crawl` |
| `robot/stair` | `Trigger` | `/robot_control/stair` |
| `robot/switch_remote` | `Trigger` | `/robot_control/switch_remote` |
| `robot/emergency` | `SetBool` | `/robot_control/soft_emergency_stop` |

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `control_rate_hz` | `20.0` | `/cmd_vel` publish rate |
| `input_watchdog_s` | `0.35` | max age of an MPC/web command before zeroing |
| `state_timeout_s` | `1.0` | max age of `RobotStatus` before zeroing (fail-closed) |
| `zero_frames` | `5` | explicit zero frames after a source change |
| `auto_take_control` | `true` | claim SDK control on startup |
| `take_control_retry_s` | `5.0` | retry interval while refused |
| `posture_timeout_s` | `10.0` | timeout waiting for posture / `switch_remote` completion |
| `estop_timeout_s` | `5.0` | timeout waiting for `soft_emergency_stop` response |
| `max_linear` | `0.0` | clamp, m/s; **0 disables** |
| `max_angular` | `0.0` | clamp, rad/s; **0 disables** |
| `cmd_vel_topic` | `/cmd_vel` | output topic |
| `state_topic` | `/robot_control/state` | input topic |

`input_watchdog_s` defaults to 0.35 s, but `m1.launch.py` sets **0.25 s** so our
watchdog expires *before* `robot_control`'s own `command_timeout_ms: 300`.

## Deliberately not done here

- **No odometry publication.** `robot_control` already publishes
  `/robot_control/odometry` (`nav_msgs/Odometry`, `odom` frame). Remap it into
  `vln_mpc`'s `odom` topic. Re-publishing would add latency and a second source
  of truth.
- **No TF.** `robot_control` owns `odom -> base_link`.
- **No camera.** The vendor stack publishes it. See below for why the transport
  must be `compressed`.

## Camera: use `compressed`, and include `/compressed` in `image_topic`

Two independent reasons to use `compressed`:

1. `vln_client`'s `raw` path rejects anything that is not `rgb8`:
   ```python
   if encoding != "rgb8":
       self.get_logger().warn(f"camera encoding must be rgb8, got {encoding!r}")
       return
   ```
   The M1 front camera publishes **`bgr8`** → every frame is dropped with a
   warning.
2. `raw` pulls in `cv_bridge`, which breaks under numpy 2.x
   (`_ARRAY_API not found`, `KeyError: 16`).

`compressed` decodes via `cv2.imdecode` and avoids both. Note that both
`vln_client` and `vln_web` subscribe directly to `image_topic` with
`sensor_msgs/CompressedImage` — they do **not** append `/compressed`
automatically:

```yaml
image_topic: /camera/front/image_raw/compressed
image_transport: compressed
```

For standalone operation independent of `zsm_console`, `m1_adapter` also ships
`m1_rtsp_camera`, which pulls the M1 front camera RTSP stream
(`rtsp://192.168.168.168:8554/front`) and publishes JPEG `CompressedImage` on
`/camera/front/image_raw/compressed` (`BEST_EFFORT` sensor QoS). `m1.launch.py`
enables it by default (`start_rtsp_camera:=true`).

## Running (standalone, SDK 0.2.3 `m1_robot_driver`)

Prerequisites, in this order:

```bash
# 1. Active chassis driver from hachimi_ws (SDK 0.2.3; old robot_control binary is deprecated)
source /home/jetson/Public/hachimi_ws/scripts/m1_driver_env.sh
ros2 launch m1_robot_driver robot_control.launch.py

# 2. Inference server (separate terminal)
cd /home/jetson/Public/LightNav-0
MODEL_PATH=$PWD/checkpoints/LightNav-0 TASK=vln bash scripts/serve_orin.sh
```

Build `robot_deploy` once with `.venv_ros` (keeps `hachimi_ws` untouched):

```bash
bash /home/jetson/Public/LightNav-0/robot_deploy/scripts/build_ros.sh
```

Then launch the VLN stack (overlays cleanly on `hachimi_ws/install_m1_driver`):

```bash
source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh
ros2 launch vln_bringup m1.launch.py
```

Verify before enabling motion:

```bash
ros2 topic echo /robot_control/state --once     # machine_status_name='remote', motion_status_name='walk'
ros2 topic echo /cmd_vel                        # zeros until a source is enabled
ros2 topic echo /diagnostics --once             # sdk_control_acquired: True, remote_ready: True
# If machine_status_name is 'idle' and the site is clear:
# ros2 service call /robot/switch_remote std_srvs/srv/Trigger "{}"
ros2 service call /control/set_auto std_srvs/srv/SetBool "{data: true}"
```

`sdk_control_acquired: False` means the phone APP (or another SDK client) holds
control — close it first.

## Tests

Pure safety rules need nothing but Python:

```bash
cd src/robot_adapters/m1_adapter
PYTHONPATH=. ../../../../.venv_ros/bin/python -m pytest test/test_safety.py -q
# 17 passed
```

End-to-end tests use a fake `robot_control` (real ROS messages, real DDS, zero
robot motion). They need `robot_control`'s messages importable:

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/Public/hachimi_ws/install/setup.bash
export PYTHONPATH=$PWD:$PYTHONPATH
../../../../.venv_ros/bin/python -m pytest test/ -q
# 27 passed
```

They are skipped automatically (via `pytest.importorskip`) when `robot_control`
is not available, so a plain `pytest` run on another machine still works.

## A note on the executor

The posture and e-stop handlers block while waiting for `robot_control`'s reply.
They are therefore registered in a `ReentrantCallbackGroup` and `main()` uses a
`MultiThreadedExecutor`. With the default mutually-exclusive group and a
single-threaded executor, the service client's future needs the very thread the
handler is sleeping on, and every call times out. This was caught by the
integration tests, not by inspection.
