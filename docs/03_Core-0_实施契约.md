# IBERO Core-0：CableTension-v0 实施契约

> 状态：已实现并通过本地测试｜2026-09-07

## 目的

Core-0 不是正式 benchmark，也不是 `cable_handover` 的完整任务。它只验证
IBERO 最小的双臂力学闭环：固定工位双臂能把一根线束拉到安全目标张力，并且
产生可审计的力、损伤和自碰撞结果。

任务 ID：`ibero/CableTension-v0`

成功条件：线束张力连续 12 个策略周期保持在 `[3, 7] N`，且没有张力损伤或
左右臂自碰撞。张力超过 `15 N` 会终止 episode 并记录损伤。

## 已冻结接口

- `reset(seed=...) -> (observation, info)`；
- `step(action) -> (observation, reward, terminated, truncated, info)`；
- 动作是 `float32[14]`，按
  `[left Δxyz, Δrpy, grip, right Δxyz, Δrpy, grip]` 排列，取值范围 `[-1, 1]`；
- 策略周期为 20 Hz，MuJoCo 物理步长为 0.002 s（每个策略周期 25 个子步）；
- `info["audit"]` 包含 `cable_tension_n`、`damage_flags`、
  `self_collision_flags` 和 `in_band_steps`。

观测字典暂时只包含可用于闭环调试的低维量：

| key | 内容 |
|---|---|
| `proprio` | 双臂 14 关节的 qpos 与 qvel，共 28 维 |
| `ee_pose` | 左右工具锚点的 3D 位置、工具 z 轴、占位标量，共 14 维 |
| `wrench` | 左右腕 link 的 `[Fx,Fy,Fz,Tx,Ty,Tz]` 外力代理，共 12 维 |
| `cable` | `[tension_N, anchor_distance_m]` |

## 物理边界

G1 和两只 Menagerie Robotiq 2F-85 被原样组合；行走用的 floating base 被删除，
其余下肢/腰关节由位置执行器保持在 reset 位姿。线束当前由两部分构成：

1. 两个工具锚点之间的 MuJoCo spatial tendon，提供真实的伸长弹簧力；
2. 一串 mocap capsule，依据锚点位置形成确定性的下垂外观。

因此 Core-0 的线束**还不是**能与夹爪/工件接触的 Flex 线缆；硬质端子也暂时
直接连接到工具锚点，而非依赖手指接触抓稳。这是有意隔离“力控 Core”与
“柔性抓取”风险的取舍，后者是 Core-1 的工作。

## 可重复性与验收

- Python 3.11、MuJoCo 3.8.1，以及仓库内的阻尼最小二乘双臂控制器；
- G1 / 2F-85 模型快照保存在 `assets/vendor/`，并随附上游许可证；
- 相同 seed 的 reset 产生相同观测；
- `pytest` 覆盖 reset 确定性、Gymnasium space/audit、脚本成功和
  `gymnasium.utils.env_checker`；
- `python -m ibero.demo --save-frame artifacts/cable_tension.png` 生成最终帧；
- `python -m ibero.demo --viewer` 打开交互式 MuJoCo 窗口。

## 明确不在本阶段

- 柔性抓取、端子接触、Flex/elasticity 标定；
- 相机、RGB-D、LeRobot、ACT、示教或遥操作；
- 通用 `scene_config.yaml` / `task_spec.py` 插件体系；
- ROS 2、多引擎后端、压装/拆解工艺。

当 Core-0 的张力传感、力方向和可视化形态经人工检查后，Core-1 才把当前参数
收敛为最小 YAML，并将工具锚点替换为实际 2F-85 指尖上的硬质端子接触。
