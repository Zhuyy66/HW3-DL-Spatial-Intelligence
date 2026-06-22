# Week3 Day5 Topic2 A-only vs ABC

| Model | Training data | Steps | D normalized Action L1 | D raw Action L1 | Formal closed-loop success rate | Direct-cameras smoke | Source |
|---|---|---:|---:|---:|---|---|---|
| A-only | splitA | 150000 | 0.658629 | 0.193360 | not run as a formal closed-loop benchmark; strict EGL was limited by the system graphics stack; see direct-cameras smoke evidence | A-only direct-cameras smoke: steps=60, moved=True, max_tcp_delta=0.264235, last_action_shape=[7]; json=topic2_act/eval/results/a_only_direct_cameras_smoke.json, log=logs/Week3_Day4/day4_a_only_closed_loop_direct_cameras.log | topic2_act/eval/results/a_only_splitD_action_l1.json |
| ABC | splitA+splitB+splitC | 150000 | 0.577993 | 0.168696 | not run as a formal closed-loop benchmark; strict EGL was limited by the system graphics stack; see direct-cameras smoke evidence | ABC direct-cameras smoke: steps=60, moved=True, max_tcp_delta=0.135650, last_action_shape=[7]; json=topic2_act/eval/results/abc_direct_cameras_smoke.json, log=logs/Week3_Day5/day5_abc_closed_loop_direct_cameras.log | topic2_act/eval/results/abc_splitD_action_l1.json |
