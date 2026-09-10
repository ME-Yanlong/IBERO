# 双端线束张力控制（Core-0.1 补强）

G1 双臂上的 2F-85 各自通过双指真实接触夹住一个动态端头。线束为带
轴向弹簧/阻尼和中心线弯曲力的 MuJoCo 1-D Flex。没有夹持 weld、mocap、
物体 teleport，也不把腕部 F/T 误当作线束内部拉力。

## 运行与审阅

```powershell
conda activate ibero-core
python -m ibero.check_scene scenes/cable_stretch
python -m ibero.demo --env cable_stretch --viewer --playback-rate 0.5
```

窗口初始等待。Enter 每次重新运行相同 seed，结束后保留画面；P 暂停/继续，
N 单步 0.05 s，R 回放，B 上一帧。四格显示总览、双手和线束区域，下方
显示真实张力曲线、目标/卸载阈值、夹持和审计。关闭 MuJoCo 主窗口结束。

```powershell
python -m ibero.demo --env cable_stretch --save-trace artifacts/stretch.npz --save-multiview artifacts/stretch.png --save-manifest artifacts/stretch.json
python -m ibero.demo --env cable_stretch --viewer --replay artifacts/stretch.npz
```

回放保存物理积分状态和逐帧审计，但不是策略 checkpoint。源码、资产、
场景或 MuJoCo 版本改变后必须重新生成；回放状态禁止继续积分，Enter 开新轮。

## P4 配置入口

| 文件/字段 | 物理或任务语义 |
|---|---|
| `scene_config.yaml / robot` | 末端选择、14 关节准备角，顺序见 `ARM_JOINTS` |
| `materials.cable.length_m` | 柔性段原始弧长，不包含端头 |
| `initial_span_m` | 两端头中心的初始距离，不是线束原长 |
| `outer_diameter_m` / `linear_density_kg_m` | 接触直径 / 线束质量 = 线密度 × 弧长 |
| `terminal` | 两端相同连接器的尺寸与单个质量 |
| `axial_stiffness_n_m` / `damping` | 全长等效 N/m 与 N·s/m；N 段时各段乘 N |
| `bending_stiffness_nm2` | EI，中心线弯曲能量的系数 |
| `minimum_bend_radius_m` | 曲率报警阈值；不是刚性限位或断裂模型 |
| `control` | 目标张力、导纳增益、速度限制、滤波、保持时间和扰动 |
| `constraints.yaml` | 不可由任务清除的超拉阈值/持续时间、碰撞许可、夹持窗口 |
| `task_spec.py` | 只读状态上的阶段与结果判定，不操作仿真模型 |

默认目标 2 N，容差 ±0.4 N。4.0–4.4 s 在中间节点施加约 0.6 N 向下外力，
seed 控制 ±10% 幅度变化，实际值写入 manifest。扰动结束后，两手持续夹持、
无当前弯曲报警、张力连续 2 s 在容差内，才成功。

导纳控制根据滤波张力改变两端相对距离，并保持双手中心与初始姿态。原始
张力超过 5 N 即进入卸载，低于 3 N 才退出；原始张力持续超过 8 N 达 0.04 s
会锁存 `cable_damage`。因此短时峰值超过 8 N 不等于已经触发损伤，报告
同时保留峰值和事件。松手故障返回 `dropped`，禁止碰撞返回 `self_collision`。

## 验证与边界

```powershell
python -m pytest -q
python -m ibero.verify_core01 --seeds 3 --output artifacts/core01_physics_review.json
```

验证包含不同分段的独立轴向加载、质量、弯曲能量梯度、悬垂支反力、六轴
F/T 坐标、三组材料闭环、外力扰动/卸载恢复以及松手/强拉故障。

目前所有参数为 provisional。模型没有扭转本构、自接触、截面应力、绝缘层
损伤或实际断裂；只模拟等效中心线响应。格式检查和保守步长门槛不等同于
任意材料/姿态的稳定性证明。改几何、EI、分段数、步长后应重跑验收；真实
工程标定仍需要力—伸长、弯曲/悬垂及接触摩擦实测。
