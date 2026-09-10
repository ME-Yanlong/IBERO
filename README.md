# IBERO Core-0.1

第一次了解项目，建议先读 [Core-0.1 版本介绍：架构、索引、材料与接触力控](docs/04_Core-0.1_版本介绍.md)。它按运行入口到物理实现逐层解释当前源码；开发决策与验收记录另见 [实施契约](docs/04_Core-0.1_实施契约.md)。

## 版本管理与开发分支

仓库：<https://github.com/ME-Yanlong/IBERO>。

- `main`：保存当前 Core-0.1 基线与后续经审阅、验收的稳定变更；“稳定”不代表已完成真实材料标定。
- `feature/core01-industrial-materials`：弹性卡扣、材料去除与铣削的后续开发分支，执行范围见 [工业机制扩展计划](docs/04_Core-0.1_工业材料与加工机制调研.md)。分支建立不代表这些功能已经实现。
- 后续功能先在开发分支提交和推送，满足计划中的回归与物理验收后，再通过 Pull Request 审阅合并到 `main`；不自动合并未完成实验。
- 源码、场景、文档及带许可证的必要模型资产纳入版本管理；本地环境、缓存与 `artifacts/` 运行产物按 `.gitignore` 排除，验收摘要和复现命令保留在文档中。

以上是协作约定，不代表 GitHub 已开启服务器端分支保护规则。

IBERO contains three deliberately narrow, executable simulator tasks:

- `ibero/CableTension-v0`: the original fixed-anchor force-control smoke test;
- `ibero/CableHandover-v0`: a contact-driven G1 + dual 2F-85 handover of a
  physically connected Flex cable terminal.
- `ibero/CableStretch-v0`: real-contact two-ended cable tension control,
  disturbance recovery and overload unloading (the default demo).

## 开发分支：工业材料扩展

上述三项是原基线。`feature/core01-industrial-materials` 另外实现了真实接触的卡扣按压拔出、带 Flex 线束的拔出放置，以及有实体孔槽、平均加工载荷和真实主轴的加工台架。卡扣/组合与加工台架已有阶段验收，具体冻结版本、失败记录和标定缺口见 [工业机制开发日志 §18](docs/04_Core-0.1_工业材料与加工机制调研.md)。这不等于最终源码的 S9 全量验收。

G1 加工工作站 `scenes/plate_milling/` 仍在 S8 调试，尚未通过完整 G8；不要把它当作稳定机器人加工 benchmark。审阅入口已实现：

```powershell
python -m ibero.demo --env latch_release --viewer
python -m ibero.demo --env harness_unplug --viewer
python -m ibero.demo --env plate_milling --shape through_hole --viewer
```

窗口初始等待，Enter 运行/重跑，P 暂停、N 单步、B 后退、R 回放。加工窗口的 C 只切换视觉切屑；回看材料使用独立模型/账本，不改变 live 仿真。当前工业细步长仿真明显慢于实时，不能用播放倍率加速物理求解。加工参数、索引和假设见 [工作站说明](scenes/plate_milling/README.md) 与 [台架说明](scenes/milling_bench/README.md)。

## Current scope

- fixed-base Unitree G1 upper body with two Robotiq 2F-85 grippers and pinch sites;
- 14D dual end-effector delta action;
- a scene-contributed recipe at `scenes/cable_handover/` with strict YAML schema
  and a MuJoCo 1-D Flex cable connected to dynamic end terminals;
- wrist F/T sites, cable tension, damage and self-collision audit;
- deterministic Gymnasium reset/replay, rollout manifests and calibration CLI;
- a contact-only scripted handover demo (no mocap grasp, weld or object teleport).

This is deliberately not yet a training benchmark, teleoperation stack, generic
scene system, or LeRobot integration.

## Setup

```powershell
conda activate ibero-core
python -m pip install -e ".[dev]"
python -m ibero.check_scene scenes/cable_handover
python -m ibero.check_scene scenes/cable_stretch
python -m ibero.demo --env cable_stretch --viewer --playback-rate 0.5
```

`--viewer` is a persistent review session: inspect the reset state, then press
Enter in either the MuJoCo window or its four-view dashboard to run one episode.
The final state remains on screen; another Enter repeats the same `--seed`.
Use `--playback-rate 0.5` for slow motion. The default recipe explicitly selects
`robot.end_effector.type: robotiq_2f85` and removes the fixed G1 hand visual
before mounting each 2F-85, so the two end-effectors are never overlaid.

Use `python -m ibero.demo --env cable_handover --save-multiview
artifacts/cable_handover_4view.png` for a headless scripted rollout and a 2×2
review frame. Add `--save-manifest artifacts/rollout.json` for reproducibility
metadata.

Press **P** to pause/resume, **N** to advance one control tick, **R** to replay
the recorded episode, and **B** to inspect the previous frame. Enter starts a
fresh episode; replay does not resume physics. Four views include contact-force
arrows, site coordinate axes, grasp state and a tension history graph.

```powershell
python -m ibero.demo --env cable_stretch --save-trace artifacts/stretch.npz --save-multiview artifacts/stretch.png --save-manifest artifacts/stretch.json
python -m ibero.demo --env cable_stretch --viewer --replay artifacts/stretch.npz
```

Replay checks scene, simulation source, asset hashes and MuJoCo version. After
changing any of these, regenerate the trace. Only the dashboard is optional:
without a graphics desktop use the non-viewer commands, not `--viewer`.

Run the complete simulator checks with:

```powershell
python -m pytest -q
python -m ibero.verify_core01 --seeds 3 --output artifacts/core01_physics_review.json
python -m ibero.calibrate --output artifacts/core01_calibration.json
```

The default handover recipe intentionally labels all cable values as
`provisional`. The Flex implementation validates the simulator chain; it does
not claim calibration to a real harness until scene YAML values are replaced
with measured or datasheet-backed values.

The current material uses actual Flex edge spring/damper forces plus an
energy-consistent centerline bending model. `axial_stiffness_n_m` and `damping`
are whole-cable equivalents; `bending_stiffness_nm2` is EI. Minimum bend radius
is an alarm, not a hard geometric constraint. The misleading `capsule_chain`
alias has been removed: an independent rigid-chain backend is not implemented.
There is no torsional constitutive law, cable self-contact or physical fracture.

See [Core-0.1 implementation contract](docs/04_Core-0.1_实施契约.md) and
[the tension recipe](scenes/cable_stretch/README.md) for acceptance evidence and
configuration semantics. These are engineering regressions, not generalization
benchmark scores or validation against a real harness.

## Licenses for bundled model assets

`assets/vendor/unitree_g1/` retains the Unitree BSD-3-Clause license and
`assets/vendor/robotiq_2f85/` retains its BSD-2-Clause license. Both snapshots
were obtained from MuJoCo Menagerie release `2026.9.0`.
