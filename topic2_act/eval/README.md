# Day 6 ACT-CALVIN Bridge

This directory contains the Day 6 connectivity path for Topic 2:

- CALVIN env and rollout run in `env_hw3_calvin_eval` / Python 3.8.
- LeRobot ACT inference runs in `env_hw3_robot` / Python 3.12.
- `LeRobotACTWrapper` keeps the CALVIN `reset()` / `step(obs, goal)` interface and delegates real inference to one persistent worker subprocess.

## Protocol

The client and worker use stdout/stdin only for framed protocol messages:

```text
4-byte big-endian payload length + pickle protocol 4 payload
```

Worker logs must go to stderr or `--log-file`, never stdout. Message types:

- Worker -> client `ready`: checkpoint path, device, feature summaries, action dim, ACT chunk settings.
- Client -> worker `reset`: clears ACT action queue and processor state.
- Worker -> client `reset_ok`: reset count.
- Client -> worker `step`: raw CALVIN obs dict plus goal. The client does not resize, normalize, or remap observations.
- Worker -> client `action`: unnormalized CALVIN action as a plain Python `list[float]` of length 7, plus action stats.
- Worker -> client `error`: message, repr, traceback.
- Client -> worker `close`; worker -> client `closed`.

## Observation and Action Contract

The worker converts raw CALVIN obs into the LeRobot batch:

- `rgb_obs.rgb_static` or top-level `rgb_static` -> `observation.images.image`
- `rgb_obs.rgb_gripper` or top-level `rgb_gripper` -> `observation.images.wrist_image`
- `robot_obs` -> `observation.state`

The checkpoint configuration defines the expected feature shapes. The current smoke500 baseline uses:

- static image: `200x200x3`
- gripper image: `84x84x3`
- state: `15`
- action: `7`

The returned worker payload is a plain Python `list[float]` of length 7. `LeRobotACTWrapper` converts it back to `np.float32[7]` on the CALVIN side. The first six values are relative EE deltas clipped to `[-1, 1]`, and the final gripper command is discretized to `-1` or `1`.

## Day 6 Acceptance

Run the fidelity check in `env_hw3_robot`:

```bash
python topic2_act/eval/run_bridge_fidelity.py \
  --checkpoint "$CKPT" \
  --dataset-root /root/Test/Zhr/DL/HW3/topic2_act/data/xiaoma26_calvin_lerobot/splitA_episodes_A_smoke500_canonical_v3 \
  --worker-python /opt/conda/envs/env_hw3_robot/bin/python \
  --device cuda:0 \
  2>&1 | tee logs/Day6/day6_bridge_fidelity.log
```

Run one CALVIN D rollout in `env_hw3_calvin_eval`:

```bash
python topic2_act/eval/run_calvin_eval.py \
  --calvin-root /root/Test/Zhr/DL/HW3/topic2_act/calvin_official \
  --dataset-path /root/Test/Zhr/DL/HW3/topic2_act/calvin_official/dataset/calvin_debug_dataset \
  --checkpoint "$CKPT" \
  --worker-python /opt/conda/envs/env_hw3_robot/bin/python \
  --worker-device cuda:0 \
  --worker-log logs/Day6/day6_worker_real_act.log \
  --eval-log-dir /root/Test/Zhr/DL/HW3/topic2_act/outputs/calvin_eval/day6_real_act_direct_cameras \
  --cuda-device 0 \
  --egl-policy direct-cameras \
  --single-rollout-smoke \
  --rollout-steps 60 \
  2>&1 | tee logs/Day6/day6_real_act_direct_camera_rollout.log
```

Pass criteria:

- `day6_bridge_fidelity.log` has `bridge_fidelity_result.exact: true`.
- `day6_real_act_direct_camera_rollout.log` has no worker protocol errors and prints `single_rollout_smoke_result`.
- `single_rollout_smoke_result.last_action_shape` is `[7]`.
- `single_rollout_smoke_result.max_tcp_delta > 1e-4`.
- `day6_worker_real_act.log` records the checkpoint path, policy device, input features, action stats, and reset count.

`direct-cameras` is a connectivity route for the current NVIDIA EGL issue. It is not the final benchmark evaluation path.
