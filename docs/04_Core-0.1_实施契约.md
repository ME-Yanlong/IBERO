# IBERO Core-0.1：从力控烟雾测试到可贡献线束场景的实施契约

初次了解平台的读者，请先阅读 [Core-0.1 版本介绍](04_Core-0.1_版本介绍.md)：其中解释目录、主程序、索引关系、材料建模与接触力控。本契约侧重开发边界、决策与验收。

> 状态：**已实现并完成软件验收；真实线束参数仍为 provisional**  
> 前置版本：`docs/03_Core-0_实施契约.md`  
> 本文作用：记录冻结后的边界、实际实现和验收结果；它不是对真实产线材料标定的声明。

> 2026-09-09 补充：当前物理实现与验收以第 14 节为准；第 12–13 节保留历史记录，
> 其中旧安装姿态、端点拉力代理与 20/20 结果不能替代新模型验证。

## 1. 重新审视后的判断

《工业级双臂具身智能操作基准平台建设方案》的方向是正确的：IBERO 最终应当是面向工业双臂操作、可复现、可评测、可贡献任务的基准平台，而不是一个单一控制 demo。方案中尤其重要的承诺是：

- 机器人、材料、任务不能各自硬编码；
- 场景贡献者应以 `scene_config.yaml + task_spec.py` 定义任务；
- 线束任务必须把接触、受力、损伤和双臂协作纳入同一条评测链；
- 物理、传感、控制、数据记录和评测需要有稳定的边界。

但这些承诺不能通过一开始就实现完整 benchmark、LeRobot 数据管线、ACT 训练、ROS2 和多场景套件来兑现。那会让我们无法判断问题究竟来自机器人构型、线束物理、任务定义还是策略。

现有 Core-0 已经完成了正确的“烟雾测试”：固定基座 G1、双臂控制、视觉上的线缆、力区间任务、Gymnasium API、可渲染 demo 和回归测试均已跑通。它也有明确的刻意简化：线缆是空间 tendon 加 mocap 胶囊的可视化近似，末端锚在腕部 site，`wrench` 是刚体外力代理。因此它**不是**真实夹爪抓取、接触驱动线束、Flex 线束，也不能作为 `cable_handover` 的物理依据。

Core-0.1 的工作不是“给 Core-0 多加功能”，而是完成一个可复用的垂直切片：

```text
P2 机械师：可验证的 G1 + 2F-85 物理构型
                 ↓
P3 材料专家：场景参数化、可接触的 1-D Flex 线束与端头
                 ↓
Core：编译、控制、传感、审计、复现的稳定接口
                 ↓
P4 菜谱作者：scene_config.yaml + constraints.yaml + task_spec.py
                 ↓
一个真实接触驱动的 CableHandover 场景与可重复 scripted baseline
```

本版本的成果应当让随后新增场景时，主要编写声明式配置和受约束的任务逻辑，而不是复制 MuJoCo 装配代码。

## 2. Core-0.1 的唯一交付目标

交付首个可贡献场景：`scenes/cable_handover/`，并注册环境 `ibero/CableHandover-v0`。

该任务的最小语义是：左 2F-85 初始夹持线束左端的刚性端头；右 2F-85 接近并夹住同一端头（或一个明确指定的接力端头）；左手释放；右手将端头送入目标接收区。全过程使用真实夹爪—端头接触，线束连续地响应两端运动，并被独立记录拉力、损伤与机器人自碰撞事件。

这里的“抓住”不能再由 wrist site weld、mocap 跟随或隐藏锚点宣称成功。为保证第一版可控，线束两端允许设置有真实几何和质量的**刚性端头/夹持帽**；夹爪先完成“夹持端头”，再逐步发展到“夹持裸线束”。这不是降低目标，而是把可验证的接触抓取和高度不确定的软体细线夹持分开验证。

`CableTension-v0` 保留为 Core-0 回归环境，不重定义为 handover，也不删除。

## 3. 范围与非范围

### 本版本必须完成

1. 一个派生、固定基座的 G1 双臂 + 双 2F-85 运行构型，具有可验证的末端、夹持、力/力矩、碰撞与相机命名接口。
2. 一个可接触、带端头的线束模型，以及其**场景级**材料参数和校准测试。
3. MuJoCo 1-D Flex/elasticity 线束实现；当场景给出材料字段时，菜谱可明确选择 `representation: flex`。
4. 场景配置加载、schema 校验、任务规则接口、约束审计、确定性 reset/replay 的 Core 骨架。
5. `cable_handover` 的 YAML 菜谱、Python 任务规则、scripted baseline、测试与可视化 demo。

### 明确不在 Core-0.1 内

- LeRobot/HDF5/Zarr 正式数据集、ACT/扩散策略训练、遥操作采集和人类演示；
- ROS2、真机驱动、硬件安全认证；
- 多物理后端、通用插件市场、公共排行榜、OOD 大套件；
- 任意 Python 可在任务文件中直接改写 MuJoCo 模型的“万能脚本”机制；
- 未经真实线束数据校准便声称“工业级高保真”。

Core-0.1 会保留未来数据记录所需的 rollout manifest 与事件记录接口，但不实现训练数据产品。

## 4. 先冻结的架构边界

### 4.1 推荐目录和所有权

```text
src/ibero/
  core/
    scene_loader.py       # YAML 读取、schema 校验、内容哈希
    scene_compiler.py     # 配置 + 资产 + preset -> MuJoCo model
    task_api.py           # TaskSpec、TaskState、TaskResult 的稳定接口
    audit.py              # 损伤、自碰撞、事件、rollout manifest
    sensors.py            # 统一的传感器读数与坐标变换
  robots/
    g1_upperbody_2f85.py  # P2 的派生运行构型与命名映射
  materials/
    cable.py              # P3 线束表示接口与参数模型
  envs/
    cable_tension.py      # 保留的 Core-0 回归环境
    cable_handover.py     # 由场景编译器构建的首个环境
  control/
    resolved_rate.py      # 保持可替换的控制器接口
scenes/
  cable_handover/
    scene_config.yaml
    constraints.yaml
    task_spec.py
    README.md
tests/
  ...
```

`src/ibero` 持有平台机制；`scenes/<id>` 持有一个可贡献任务的声明与语义。场景不得反向依赖某个环境的私有字段，也不得复制机器人或材料装配逻辑。

### 4.2 三层契约

| 层 | 负责什么 | 不负责什么 |
|---|---|---|
| Core | 校验、编译、控制循环、传感器归一化、审计、reset/replay、渲染和注册 | 指定某一任务的成功语义或材料参数 |
| P2/P3 资产提供者 | 机器人 preset、材料模型、坐标/关节/碰撞映射、参数及其测试 | 把任务成功规则硬编码到资产中 |
| P4 菜谱作者 | 场景实例、随机化范围、任务阶段、成功/失败谓词与文本说明 | 直接操控 `MjModel`、覆盖 Core 安全审计 |

### 4.3 场景文件的职责分离

`scene_config.yaml` 是静态、声明式的“摆什么、用什么、如何初始化”。最小字段如下：

```yaml
schema_version: ibero.scene/v0.1
id: cable_handover
suite: harness
backend: mujoco
robot:
  preset: g1_upperbody_v0
  end_effector:
    type: robotiq_2f85
physics:
  timestep_s: 0.002
  control_hz: 20
materials:
  cable:
    representation: flex
    length_m: 0.48
    outer_diameter_m: 0.010
    linear_density_kg_m: 0.055
    minimum_bend_radius_m: 0.045
    terminal: {length_m: 0.24, width_m: 0.014, height_m: 0.018, mass_kg: 0.060}
    parameter_source: {length_m: provisional}
initialization:
  seed_range: [0, 9999]
  target_jitter_m: [0.010, 0.010, 0.004]
sensors:
  required: [left_wrist_ft, right_wrist_ft, cable_tension, safety_events]
render:
  cameras: [front, left_wrist, right_wrist]
```

`constraints.yaml` 放置强制安全和成功相关阈值，例如线束拉力上限/持续时间、允许的碰撞对、目标区域、时间上限和 reset 合法区间。其值可以被 `task_spec.py` 读取，但不能被 task 覆盖。

`task_spec.py` 只描述任务语义。它实现受限的 `TaskSpec`：从只读 `TaskState` 读取命名对象状态、夹持状态、接触事件、规范化 F/T、线束拉力和审计事件，输出 `TaskResult`（奖励、阶段、成功、失败原因）。它不直接导入或修改 MuJoCo model/data，不创建资产，不以随机副作用改变 reset。这样任务规则仍可在未来迁移到其他后端。

首个任务需要的谓词至少包括：`grasped_by(gripper, object)`、`released_by(...)`、`in_region(object, region)`、`cable_tension_below(limit)`、`no_damage()`、`no_disallowed_self_collision()` 和有序的 handover 阶段状态机。

## 5. P2：机械师交付契约（G1 裁剪 + 双臂 + 夹爪）

### 5.1 运行构型

P2 交付的是有版本的派生 preset：`g1_upperbody_v0`，并由 `robot.end_effector.type` 选择末端构型。当前实现以可复现的 `MjSpec` builder 组合 Menagerie 资产：删除 floating root、删除左右 hip 根及其全部下肢 subtree、删除因裁剪而失效的 locomotion keyframe；保留 pelvis、腰部、躯干和双臂。腰部三个关节不暴露给任务 action，而由具名的 persistent hold target 固定在 reset 位姿。这个裁剪/锁定策略在代码、model handles 和测试中是显式的，不是静默地以实时当前位置抵消。

默认的 `robotiq_2f85` 在挂载外接 2F-85 前移除源 G1 模型的两个固定 `rubber_hand` 视觉 geom，因此不存在“原手 + 夹爪”叠加。`g1_native_hand` 则保留这两个静态外观、不挂 Robotiq，供构型检查使用；该 vendored G1 资产不含可驱动手指、指垫碰撞或灵巧手动作映射，故不能把该选项标为已支持的灵巧手 handover。真实 G1 灵巧手应在导入相应资产后另建末端类型、动作接口和接触回归。

该 preset 必须公开稳定名称：

- 左右 7-DoF 臂的关节顺序、限位、控制范围与 action 映射；
- 左右 2F-85 根 body、TCP、指尖接触 geom、夹持参考 site；
- 左右 wrist F/T site；
- `front`、`left_wrist`、`right_wrist` 三个相机；
- 机器人 link/geoms 的碰撞组与允许碰撞对；
- 供 P4 使用的 `left_gripper`、`right_gripper`、`left_tcp`、`right_tcp` 名称。

夹爪必须以真实开合 joint 和指尖 geom 接触端头。任务中允许 scripted baseline 使用接近—闭合—抬升的动作序列，但不允许把被抓物体 teleport、weld 到 TCP 或转为 mocap。

### 5.2 传感与碰撞定义

- F/T 读数统一为各 wrist sensor frame 下的 `[Fx, Fy, Fz, Tx, Ty, Tz]`，单位为 N/N·m；Core 同时提供变换到世界系的只读版本，并在观测 metadata 中写入 frame。
- 现有 `cfrc_ext` wrist-link 代理在 `CableTension-v0` 可继续存在，但不得作为 `CableHandover-v0` 的正式 F/T 数据。
- 自碰撞是“左右机械臂及躯干之间所有**未允许**的机器人 geom 接触”。相邻 link、同一夹爪内部、明确的装配接触对须列在 allowlist；线束接触机器人不属于自碰撞。每个被忽略的 pair 需要理由和测试。

### 5.3 P2 验收

1. 静止 10 秒无 NaN、无未预期漂移、无未允许自碰撞。
2. 两臂可分别到达规定 handover 区的测试位姿，关节不越限。
3. 两个 2F-85 能够各自对刚性校准端头完成闭合、提起和保持；不能使用焊接/teleport。
4. 已知外载荷的静态测试中，F/T sign、单位和 frame 经验证；模拟基准误差目标不高于 5%。
5. 每个机器人名称、自由度、碰撞 allowlist 和安装变换都有自动化检查。

## 6. P3：材料专家交付契约（线束与 Flex）

### 6.1 先建立可替换的材料接口

线束材料直接定义在每个场景的 `scene_config.yaml`：长度、外径、线密度、最小弯曲半径、端头质量/尺寸、弯曲刚度、轴向刚度、阻尼、接触摩擦、随机化范围和参数来源。每个值必须注明 `measured`、`datasheet` 或 `provisional`，以避免把仿真默认值误当材料事实。这样每一个贡献菜谱都能拥有自己的线束，而无需修改 P3 代码。

Core 只依赖 `CableModel` 能提供的命名端头、线束状态、拉力、损伤输入和碰撞几何；不依赖某一种具体 MuJoCo 表达。

### 6.2 选择策略与已实现的 Flex

本节以下为早期实现记录。第 14 节已废止 `capsule_chain` 兼容别名，并替换旧拉力代理；当前只验收独立命名的 Flex 实现。

用户确认后的策略是：材料参数属于场景；当菜谱提供这些字段且需要柔性响应时，菜谱直接选择 `representation: flex`。`cable_handover` 已选择 MuJoCo 3.8.1 的 1-D Flex：它由连续的可碰撞 capsule 元素构成，两端以 `connect` 物理约束连接到动态端头；夹爪与端头之间没有 weld、mocap 或 teleport。

`representation: capsule_chain` 仍是可选的低保真标签，复用相同的 1-D 接触 capsule 与端头接口，以保证贡献场景不需要改写控制、传感或任务代码。它不是“没有 Flex 的视觉替身”。

软件验收已完成：20 个固定 seed、每个 10 秒安全保持均无 NaN/约束爆炸；20 个 handover baseline 均成功。物理保真验收则分开处理：只有在用 `measured`/`datasheet` 参数完成拉伸与悬垂对照后，才能声称该场景与真实线束的力—伸长或曲率一致。

### 6.3 受力与损伤

`cable_tension` 是线束自身的轴向拉力，不可用腕部 F/T 代替。损伤由 Core 审计器根据 P3 提供的拉力信号和 `constraints.yaml` 中的“阈值 + 超限持续时间”判定；一旦触发，事件不可由任务代码清除。正式任务的 failure reason 必须区分 `cable_damage`、`self_collision`、`timeout` 与 `task_condition_unmet`。

Core-0.1 不把“协同内部力”作为官方成功指标：对柔性线束，先报告可解释的 `cable_tension` 与双腕 F/T；刚体协同操作的内部力定义留给后续对象套件。

### 6.4 P3 验收

1. 同一参数、同一 seed 的线束 reset 和前 10 秒轨迹可重复。
2. 胶囊链在夹爪、端头、桌面/接收区的接触下无 NaN、无穿透失控、无 mocap 跟随捷径。
3. 拉伸、悬垂、端头质量三个小型校准脚本均生成机器可读报告。
4. 线束拉力、损伤事件和参数来源写入 rollout manifest。
5. Flex 运行必须输出稳定性、F/T 静载、seed 与参数来源报告，而不是“看起来像软线”。

## 7. P4：菜谱作者交付契约

P4 的第一个场景目录是 `scenes/cable_handover/`。它应当让一位不修改 Core 的作者能够读懂并复现实验。

`scene_config.yaml` 定义 robot preset、线束资产和表示、环境静态物体、相机、物理步长、随机化字段与场景 ID；`constraints.yaml` 定义不可绕过的阈值和允许碰撞；`task_spec.py` 定义阶段机和结果判定；`README.md` 说明物理假设、已知限制和命令。

首个 handover 阶段建议冻结为：

1. `left_holding`：左夹爪已确认夹持指定端头；
2. `right_approach`：右 TCP 进入安全接近区；
3. `dual_grasp`：右夹爪确认夹持，左夹爪仍持有；
4. `left_release`：左夹爪释放且右夹爪持续夹持；
5. `place`：右端头进入目标接收区；
6. `success`：满足 place、无损伤、无禁止自碰撞、全过程未越过安全约束。

对“夹持确认”的最低要求是：指尖—端头接触存在、夹爪开度处于闭合范围、端头相对夹爪的短窗口位移低于阈值。仅接触或仅闭合均不算抓住。

P4 的随机化先限制在可解释范围：线束初始 yaw/平移、目标区平移、端头小姿态扰动。随机化范围必须写在 YAML，且每次 rollout 将 resolved seed 和 resolved config hash 写入 manifest；不在任务代码内暗藏随机数。

### P4 验收

1. `scene_config.yaml` 和 `constraints.yaml` 可被 schema 校验，未知字段或非法引用会失败；
2. 同一 scene ID、内容哈希、seed 能重建相同初始状态；
3. `task_spec.py` 在不接触 MuJoCo 私有对象的情况下覆盖成功、损伤、自碰撞、超时和阶段倒退；
4. 场景能通过一个检查命令（建议 `python -m ibero.check_scene scenes/cable_handover`）；
5. README 清楚说明场景所选 representation、参数来源和真实标定限制。

## 8. Core 接口与审计契约

Core-0.1 只建设一套足以承载这个场景的窄接口，而不是抽象过度的通用引擎：

```text
SceneLoader.validate(path) -> ValidatedScene
SceneCompiler.compile(validated_scene, seed) -> CompiledScene
TaskSpec.reset_context(seed, resolved_scene) -> TaskContext
TaskSpec.evaluate(readonly_task_state, audit_snapshot) -> TaskResult
Auditor.observe(model, data) -> events / metrics
```

`TaskState` 至少有命名 body/site pose、关节状态、夹爪开度、命名接触、规范化 F/T、线束拉力、阶段时钟；`AuditSnapshot` 至少有损伤锁存、自碰撞事件、超限峰值、已解析 seed、config/asset hash。`TaskResult` 至少有 reward、stage、terminated、truncated、failure_reason、success，以及机器可读 metrics。

每次 rollout 输出轻量 manifest（JSON 或等价结构），至少包含：IBERO 版本、MuJoCo 版本、scene/config hash、asset hash、材料表示与参数、seed、控制频率、动作接口版本、结果与安全事件。它是未来接入 HDF5/Zarr/LeRobot 前的复现实验凭据，不是数据集格式承诺。

环境继续遵循 Gymnasium：`reset() -> (obs, info)`，`step() -> (obs, reward, terminated, truncated, info)`。Core-0 的旧环境需要保持此行为，避免因重构破坏回归。

## 9. 按依赖排序的开发清单

以下顺序是实施顺序，不是并行角色图。每一步完成并通过本步验收后再进入下一步；P3 Flex spike 可在第 4 步之后与 P4 编写并行，但不改变集成主线。

| 步骤 | 产出 | 主要责任 | 退出条件 |
|---|---|---|---|
| 0 | 本契约审阅并冻结关键阈值/命名 | 全体 | 已确认；未决物理参数标为 provisional |
| 1 | `core/` 最小接口、scene schema、场景检查命令骨架 | Core | 能校验一个空但合法的 scene；Core-0 测试不回退 |
| 2 | `g1_upperbody_v0` 派生 preset、末端选择、命名映射、碰撞策略、F/T sites、三相机 | P2 | 通过 P2 静止、可达、夹持、F/T 静态测试 |
| 3 | SceneCompiler 将 YAML、preset、资产编译为可运行 MuJoCo 场景 | Core + P2 | 无 handover 专用硬编码即可加载 robot 和静态目标区 |
| 4 | `CableModel`、场景级 cable 字段、接触 1-D Flex、拉力/损伤校准工具 | P3 | 20 seed × 10 秒稳定性通过；参数来源被记录 |
| 5 | Flex/elasticity 接入与报告 | P3 | 已完成软件稳定性/F-T/基线报告；真实材料对照待数据到位 |
| 6 | `cable_handover` 的 YAML、constraints、受限 `TaskSpec`、阶段谓词 | P4 + Core | 场景检查通过；任务逻辑可在 mock TaskState 单测 |
| 7 | `CableHandoverEnv`、标准观测/动作、审计器、manifest 和受控 reset/replay | Core | 同一 seed 可复现；失败原因可区分 |
| 8 | 接触式 scripted baseline（接近、双夹持、释放、放置） | P2 + P3 + P4 | 20 固定 seed 上执行并产生结果表 |
| 9 | 回归测试、性能基线、可视化 demo、场景 README、开发者文档 | Core | 满足第 10 节 DoD |

实施中严禁为了跨过步骤 8 而加入 hidden weld、mocap 抓取、直接修改物体 pose 或任务特例成功判定。若 baseline 不稳定，应先定位为 P2 接触、P3 参数、场景初始化或控制问题。

## 10. Core-0.1 完成定义（Definition of Done）

只有同时满足以下条件，Core-0.1 才可结束并转入“多场景/数据/策略”阶段：

1. `CableTension-v0` 的既有回归测试仍全部通过；
2. `ibero/CableHandover-v0` 由 `scenes/cable_handover/` 编译得到，而非单文件硬编码场景；
3. G1 双臂、两个 2F-85、F/T sites、三相机、碰撞 allowlist 都有稳定的公开名称和测试；
4. handover 使用真实指尖—端头接触；没有腕部锚点、weld、teleport 或 mocap 抓取捷径；
5. 默认线束模型通过 20 个固定 seed、每个至少 10 秒的稳定性检查，并产出拉伸/悬垂/端头质量校准报告；
6. `TaskSpec` 能正确给出成功、`cable_damage`、`self_collision`、`timeout` 和未完成任务的区别；
7. scripted baseline 在 20 个固定且记录的 seed 上达到至少 18 次成功，且成功回合均无损伤、无禁止自碰撞；
8. 每次运行都有 config/asset hash、seed、参数和审计事件的 manifest；
9. Flex 有可复现实验报告；菜谱只在显式声明材料字段和参数来源时选择 `representation: flex`；
10. 非交互命令可生成一张或一段可视化结果，方便人工检查“确实是夹爪传递端头，而非视觉假象”。

第 7 项是工程回归门槛，不是泛化 benchmark 分数；它只证明默认构型和 scripted baseline 可重复。

## 11. 你需要提前准备的内容

在开始步骤 1 前并不需要你额外下载软件：当前 Python/MuJoCo/Gymnasium/G1/2F-85 资产已经足够启动。真正会影响 P3 物理可信度的是线束资料。若你能提供，优先级如下：

1. 目标线束的 datasheet 或实测：长度、外径、单位长度质量、最小弯曲半径、端头尺寸/质量、允许拉力或破坏/报警阈值；
2. 线束外皮材料及与 2F-85 指垫、桌面/接收槽的摩擦信息（没有也可先用 provisional 值）；
3. 若最终夹持对象不是通用圆柱端头，而是特定连接器：其 CAD、尺寸图或至少正侧面尺寸；
4. 若有真实 G1 安装架、手腕转接板或相机型号：其坐标/尺寸。没有时以现有仿真安装位姿作为基准并标记为仿真构型。

缺少这些资料不会阻塞 Core-0.1 的软件架构、接触验证和 Flex；它只限制“参数已校准到真实产线线束”的结论。资料到位后应作为版本化的场景 YAML 字段（附来源说明）进入仓库，而不是散落在代码常量中。

## 12. 已确认决策与验收记录

已确认：首任务夹持刚性端头；材料字段由场景定义，字段具备时可以选择 Flex；LeRobot/演示采集/策略训练和更大 benchmark 后置。

本次实现的可复现实测（MuJoCo 3.8.1、固定 seed 0–19）为：Flex 安全保持 **20/20** 在 10 秒内稳定；contact-only scripted handover **20/20** 成功、无损伤和禁止自碰撞；重力关闭的 5 N wrist F/T 合成静载检查相对误差 **0.11%**。报告由 `python -m ibero.calibrate` 生成。所有线束数值仍标为 `provisional`，因此这些结果证明的是仿真软件链路，不是产线材料标定。

## 13. Core-0.1 持续开发日志（2026-09-08）


这不是新的 benchmark 范围，而是对人工审阅、P2 构型表达和源码可审阅性的补强。下列四项已实现；所有图形窗口的最终主观观感仍应由实际开发桌面审阅。

| 编号 | 审阅发现 | 决策与实现边界 | 验收方式 | 状态 |
|---|---|---|---|---|
| UX-01 | `--viewer` 的 scripted rollout 会在一轮结束后立即关闭，人工还来不及检查运动与接触。 | 已改为持久审阅会话：初始暂停，焦点位于 MuJoCo 窗口或信息面板时按 Enter 执行一轮；结束后保留最终状态并等待下一次 Enter 或用户关闭窗口。同一 `--seed` 每轮重建，便于可重复比较。 | 代码路径已实现；需在实际图形桌面连续按两次 Enter 做最终人工确认。 | 已实现，待图形端验收 |
| UX-02 | 单一 MuJoCo 视角不足以审阅双手、端头和线束状态。 | 已实现四视角信息面板：总览、左末端、右末端、端头/目标区；显示 stage、双侧夹持、瞬时/峰值拉力、损伤和碰撞标志。桌面 GUI/Pillow 不可用时降级为保留 MuJoCo 窗口及 Enter 交互。 | `CableHandoverMultiView` 自动测试生成 640×480 的四格图；非交互 demo 已产出 PNG。 | 已验证 |
| P2-01 | 现有构型把 G1 自带手部外观与额外 Robotiq 2F-85 叠加，末端语义和视觉均不正确。 | 场景 YAML 显式选择 `robot.end_effector.type`。默认 `robotiq_2f85` 删除源资产中的两个固定 `rubber_hand` visual geom 后再挂夹爪；`g1_native_hand` 保留静态外观且不挂 Robotiq。当前资产并不含可驱动灵巧手，Core-0.1 baseline 只对 2F-85 验收。 | 自动测试确认默认模型不使用两个原生手 mesh；原生手配置没有 2F-85 body/actuator。 | 已验证 |
| DEV-01 | 核心物理、控制和配置选择缺少足够中文语义标注，不利于源码 review。 | 已在场景 YAML、P2/P3 builder、控制器、环境、四视角模块和 demo 的关键函数、超参数与可选分支补充中文设计注释。 | 相关源码可直接追溯“为什么”而非只读取 API 表面含义。 | 已完成 |

本轮自动化结果：`python -m pytest -q` 为 **12 passed**；`python -m ibero.check_scene scenes/cable_handover` 通过；`python -m ibero.calibrate` 重跑后 Flex 安全保持为 **20/20**、接触式 baseline 为 **20/20**、5 N F/T 静载相对误差为 **0.11%**；非交互 handover 仍成功，并已生成四视角 review PNG 与 manifest。报告位于 `artifacts/core01_after_p2_ux_revision.json`（构建产物，不纳入源码）。

## 14. 物理与力控补强实施记录（2026-09-09）

### 14.1 范围与实施顺序

本轮是用户批准的 Core-0.1 进一步补充，不提前引入 LeRobot、训练流水线或大规模 benchmark。目标从“脚本显示成功”推进到“安装正确、真实接触、材料有可解释力学、力控能应对扰动、失败可审计”。

| 顺序 | 已完成开发 | 主要代码与配置 | 检查方式 |
|---|---|---|---|
| 1 / P2 | 修正工具 +Z → G1 腕部 +X 的 90° 安装旋转；保留末端选择；屈肘准备位姿；移除原生手外观叠加 | `robots/g1_upperbody_2f85.py`、场景 `robot.arm_qpos` | 两侧轴向变换、末端构型、真实双指夹持、关节范围检查 |
| 2 / P3 | 实际 Flex 边弹簧/阻尼传力、能量一致的中心线弯曲、弧长和曲率观测 | `materials/cable.py`、`mechanics.py`、`bench.py` | 独立轴向试验、分段对比、质量、梯度/守恒和悬垂试验 |
| 3 / Core | 连续夹持确认、滑落失败、躯干/跨臂/工装碰撞、时间戳与损伤锁存、数值异常立即拒绝 | `envs/cable_handover.py`、`core/audit.py` | 松手、强拉、工装碰撞故障；六轴 F/T 合成载荷 |
| 4 / 力控 | 新建 `ibero/CableStretch-v0`，双端夹持、导纳拉紧、保持、扰动恢复及超限卸载 | `control/tension.py`、`scenes/cable_stretch/task_spec.py` | 2 N 目标闭环、扰动后连续保持、强扰动触发卸载再恢复 |
| 5 / P4 | 场景拥有材料、初始姿态、工装与张力控制参数；拒绝非法数值/未知字段，增加保守步长筛查 | 两个场景 YAML、`core/scene_loader.py` | 两个场景检查；三组材料 × 三种子闭环验收 |
| 6 / 审阅 | 持久窗口、四视图、暂停/单步、上一帧/回放、接触力/坐标轴、张力曲线及 NPZ 轨迹 | `review.py`、`multiview.py`、`demo.py`、`verify_viewer.py` | 真实桌面事件循环完成两轮运行及交互测试；离屏四视图与轨迹往返测试 |

源代码路径均相对于 `src/ibero/`，场景路径相对于项目根目录。关键函数、材料公式、超参数和可选分支已补充中文解释；源码通过格式与静态检查。

### 14.2 本轮纠正的关键物理语义

1. **安装**：原模型只有平移，没有把 2F-85 工具轴与 G1 腕部轴对齐。现在挂载四元数为 `[√0.5, 0, √0.5, 0]`；在两侧实际准备姿态下，工具 +Z 用腕部坐标表达均为 `[1,0,0]`。转接平移仍为仿真假设，不代表真实转接板标定。
2. **材料原长**：`length_m` 是柔性段弧长；`initial_span_m` 是端头中心距。端头边缘与 Flex 节点通过物理 connect 连接，夹爪与端头没有 connect/weld/mocap。松弛初始化按弧长生成，不把中心距当原长。
3. **轴向材料**：全长等效刚度 `k`（N/m）和阻尼 `c`（N·s/m）在 N 段串联中分别映射为 `N*k`、`N*c`。报告的 `cable_tension_n` 是实际材料边的最大正轴向力，另报两端边力；腕部 F/T 单独保留。旧具名 tendon 仅测距离，刚度/阻尼为零，不再施力。
4. **弯曲材料**：`bending_stiffness_nm2` 是 EI，使用 `E = EI/(2h) Σ|t[i+1]-t[i]|²` 的负梯度施加节点弯曲力。每个积分子步先更新当前几何，修复使用滞后一帧位置产生非物理增能的问题。`minimum_bend_radius_m` 是曲率报警，不是硬限位。
5. **表示名称**：废止旧 `capsule_chain` 别名，因为它实际上仍调用同一个 Flex 后端。当前唯一验收表示是 `flex`；独立刚体链没有实现，不能称作已有稳定回退。
6. **抓取与失败**：相对的两组指垫均接触正确端头、开度/空间窗口合格，连续三帧且相对位移小于 4 mm 才确认夹持。四个控制采样持续失去双手接触可触发 `dropped`。损伤按物理子步的阈值与持续时间锁存；碰撞事件记录实际 body 对。

材料有效刚度、直径、线密度、端头尺寸/质量、摩擦和报警/损伤阈值均由菜谱配置。更改几何并不保证原 scripted baseline 自动可达；P4 必须重新检查准备位姿、工具净空和闭环成功率。`task_spec.py` 是可信贡献者的任务代码，文本检查不是针对恶意 Python 的安全沙箱。

### 14.3 双端张力任务的明确契约

默认柔性段长 0.26 m、外径 10 mm、线密度 0.055 kg/m；两个端头各长 80 mm、质量 35 g。所有数值仍为 provisional。

- 物理步长 0.0002 s，控制周期 0.05 s。数值异常（包括 MuJoCo 自动恢复之前的 BADQACC 警告）使回合失效，不静默继续。
- 目标张力 2 N，容差 ±0.4 N。导纳反馈调整双端间距，两臂各承担一半位移，并控制共同中心和工具姿态。
- 4.0–4.4 s 对线束中间节点施加向下约 0.6 N 的实际外力，幅度由 seed 随机化 ±10%，已解析外力写入 manifest。
- 扰动结束后，两手持续夹持、无当前弯曲报警、真实张力连续 2 s 落在容差内才成功；不是滤波值或某一帧达标就成功。
- 张力超过 5 N 进入卸载，低于 3 N 才解除；超过 8 N 持续 0.04 s 触发不可清除的 `cable_damage`。短时峰值大于 8 N 但没有持续足够时间，不等于损伤，峰值仍如实记录。

### 14.4 交接任务的当前边界

校准长端头为 0.36 m，提供分离夹持区域。流程为左持有 → 双手持有 → 左释放 → 右手移至托台上方并减速 → 右张开并沿工具轴退出 → 端头由真实接触稳定支撑。

原设定的低位释放点在保持工具姿态时不可达；抬高整个托台又会撞到夹爪连杆。本轮保留低托台并使用约 0.10 m 上方的可达释放点，有限距离退出防止端头挂在连杆上。**这是重力辅助落料，不是已经实现的精密轻放或插接。** 成功必须满足双手释放、右夹爪张开、端头在 4 cm 目标半径内、真实支撑且速度小于 0.04 m/s 连续 0.5 s。

固定 seed 0–19 的新模型基线为 **18/20**，达到原契约 ≥18/20 的工程门槛，不宣称全通过：seed 11 已落在托台但位置误差约 4.80 cm；seed 14 停在接近阶段，右手尚未形成抓取。这两次均超时，无损伤/禁止碰撞。没有增大目标容差或伪造抓取来隐藏失败。

### 14.5 可复现检查结果

| 检查 | 当前结果 | 证据 |
|---|---|---|
| 自动化回归（含 Core-0） | 25 passed | `python -m pytest -q` |
| 场景与静态检查 | 两个场景通过；Ruff 检查通过 | `ibero.check_scene`、`ruff check src tests scenes` |
| 10 s 安全保持，seed 0–19 | 20/20，无数值异常/任务失败 | `artifacts/core01_stability_20seeds.json` |
| 实际接触交接，seed 0–19 | 18/20；失败明细保留 | 同上 |
| G1 腕部 5 N 增量载荷 | 幅值误差约 1.68%，符号正确 | 同上 |
| 独立轴向加载，6/12/24 段 | 预期 1.8 N，最大相对误差约 0.063%；质量与线密度×长度一致 | `artifacts/core01_physics_review.json` |
| 弯曲能量梯度与守恒 | 梯度误差约 2.1×10⁻¹¹；净力/净力矩近零 | 同上 |
| 两端固定悬垂 | 支反力与自重误差最大约 3.58%；提高 EI 后下垂减小 | 同上 |
| 六轴合成载荷 | 力、力矩、坐标旋转与力臂误差 <10⁻⁶ | 同上 |
| 默认/较软/较重材料 × seed 0–2 | 9/9，扰动后保持成功 | 同上 |
| 强扰动卸载恢复 | 触发卸载并恢复成功；峰值约 8.95 N，未达到连续损伤时间 | 同上 |
| 松手/强拉/工装碰撞故障 | 分别被判定 dropped / cable_damage / self_collision | 物理验收报告及单元测试 |
| 真实窗口事件循环 | 初始等待、完成两轮、末态保留、P/N/R/B 均通过 | `artifacts/viewer_smoke.json` |
| 轨迹与可视化 | NPZ 状态往返、回放禁止继续积分；PNG/manifest 已输出 | `artifacts/stretch_trace.npz`、`stretch_fourview.png`、`stretch_manifest.json` |

上述解析/合成试验证明实现与定义的等效模型一致，**不是与真实线束对照的材料验证**。历史 20/20 不再用于本轮结论。构建产物位于忽略跟踪的 `artifacts/`，可用以下命令重建：

```powershell
conda activate ibero-core
python -m pytest -q
python -m ibero.check_scene scenes/cable_handover
python -m ibero.check_scene scenes/cable_stretch
python -m ibero.verify_core01 --seeds 3 --output artifacts/core01_physics_review.json
python -m ibero.calibrate --output artifacts/core01_stability_20seeds.json
# 此项会显式打开实际桌面窗口；不在无头测试中自动执行。
python -m ibero.verify_viewer
```

### 14.6 直接审阅与交付

```powershell
python -m ibero.demo --env cable_stretch --viewer --playback-rate 0.5
python -m ibero.demo --env cable_handover --viewer --playback-rate 0.5
python -m ibero.demo --env cable_stretch --save-trace artifacts/stretch_trace.npz --save-multiview artifacts/stretch_fourview.png --save-manifest artifacts/stretch_manifest.json
python -m ibero.demo --env cable_stretch --viewer --replay artifacts/stretch_trace.npz
```

Enter 开始新一轮；P 暂停/继续；N 推进一个控制周期；R 从头回放当前记录；B 回看上一帧。结束后不退出窗口。四视图显示接触力和 site 坐标，面板显示夹持、张力曲线、阈值与审计。GUI 渲染会影响墙钟速度，`playback-rate` 不是硬实时保证；四格面板可关闭，不影响物理状态。

NPZ 保存积分状态与逐帧结果，禁用 pickle；回放校验场景、源码、资产和 MuJoCo 版本。回放只恢复物理画面与记录，不恢复策略内部状态，故必须 reset/Enter 才能继续新仿真；保存回放 manifest 使用该帧审计，不混入另一轮终态。修改源码/场景后须重新生成轨迹。

### 14.7 保留的限制与后续优先级

本轮批准的补强已实现，但不把以下事项包装成已完成：真实转接板/线束标定；扭转本构；线束自接触；截面应力；物理断裂；可驱动 G1 灵巧手；精密轻放/插接；任意材料与工装的自动规划。

下一轮优先顺序应是：用实测材料数据校准等效模型 → 以工具/工装可达性和接触法向力控制替代重力辅助落料 → 让双端任务覆盖更多长度/端头与姿态 → 在独立验证后加入扭转/自接触。LeRobot 与策略训练仍后置，不以接口铺开代替仿真器质量提升。

## 15. 工业材料与加工机制扩展计划索引（2026-09-10，待审阅）

后续讨论将新增重点转向弹性卡扣解锁与材料去除。详细执行清单追加在 [《Core-0.1 工业材料与加工机制调研》第 13—17 节](04_Core-0.1_工业材料与加工机制调研.md#13-core-01-工业机制扩展详细执行任务计划待审阅)，计划编号 `C01-IND-PLAN-1`。

计划包含 S0—S9 共 60 项主线任务，以及 X-01—X-08 共 8 项需单独确认的条件扩展；列明依赖、文件落点、量化门槛、证据要求和连续工作规则。详细文档是该轮任务状态的唯一登记来源，本节只作入口，不复制一份独立清单。

本次仅编写计划，尚未获准实施；没有新增仿真能力，也没有执行新增验收。第 14 节的历史完成记录不因此改变。审阅通过后再按阶段追加实际实施结果，不将研究方案或待办任务标为已完成。
