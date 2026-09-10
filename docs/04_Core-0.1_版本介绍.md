# IBERO Core-0.1 版本介绍：从运行入口到材料、接触与力控

> 面向第一次接触项目的读者。依据 2026-09-09 的项目源码编写，介绍已经实现的系统，而不是未来架构设想。
>
> 本文回答“是什么、在哪里、怎么连接、为什么这样实现”；[实施契约](04_Core-0.1_实施契约.md)回答“承诺了什么、验收了什么、还有什么限制”。两份文档互相补充，不替代彼此。
>
> 工业扩展阅读提示：第 1—18 节保留原线束基线的介绍。开发分支后来增加了卡扣、带线束拔出与端铣加工，见[第 19 节](#part-19)。该节描述已经存在的实现，但 G8/S9 的发布验收状态以[工业机制开发日志](04_Core-0.1_工业材料与加工机制调研.md#16-连续工作规则状态登记和异常处理)为准，不把开发入口当成稳定 benchmark。

## 阅读导航

建议第一次按顺序读，不必先掌握 MuJoCo。公式部分只需理解物理含义，不必推导。

- [1. 先认识现在的平台](#part-1)
- [2. 五分钟运行与观察](#part-2)
- [3. 项目目录地图](#part-3)
- [4. Core 究竟由哪些部分组成](#part-4)
- [5. 一条命令如何找到场景并启动](#part-5)
- [6. 引擎、模型与动态状态](#part-6)
- [7. 名称、ID、数组地址：如何索引仿真对象](#part-7)
- [8. P2 机械师：G1 和夹爪怎么组装](#part-8)
- [9. P3 材料专家：线束究竟怎么模拟](#part-9)
- [10. 接触、抓取和三种不同的力](#part-10)
- [11. 双臂如何执行张力控制](#part-11)
- [12. 一个 reset 和一个 step 内部发生什么](#part-12)
- [13. P4 菜谱作者与任务裁判](#part-13)
- [14. 观测、动作和运行结果](#part-14)
- [15. 可视化、轨迹与复现](#part-15)
- [16. 如何知道实现是可信的](#part-16)
- [17. 按问题定位文件与扩展项目](#part-17)
- [18. 推荐源码阅读路线与理解检查](#part-18)
- [19. 工业扩展：弹性卡扣、真实孔槽与加工载荷](#part-19)

<a id="part-1"></a>
## 1. 先认识现在的平台

### 1.1 一句话理解

IBERO Core-0.1 是一个“由场景文件配置、用真实仿真接触操作柔性线束、能够检查受力与任务结果”的双臂仿真最小平台。

这里的“真实接触”指：在仿真中，夹爪通过碰撞接触和摩擦带动动态物体，不是把物体位置强制绑定到手上。它不表示已经做过真实机器人或真实线束实验。

最重要的闭环是：

```text
场景规定机器人、材料和目标
          ↓
控制器让双臂与夹爪运动
          ↓
接触力带动端头，端头带动柔性线束
          ↓
引擎计算形变、运动和受力
          ↓
控制器读取张力并调整动作
          ↓
审计器检查安全，任务裁判判断成功或失败
```

### 1.2 现在到底有几个场景

从研究方向看，现在仍是同一个“G1 双臂操作线束”的小型场景族，并不是一个大型任务集合。从代码入口看，要区分三个名字：

| 命令中的名称 | `ibero.make()` 中的环境 ID | 当前含义 |
|---|---|---|
| `cable_tension` | `ibero/CableTension-v0` | Core-0 遗留烟雾测试：腕部锚定、简化线缆与力区间控制 |
| `cable_handover` | `ibero/CableHandover-v0` | Core-0.1：左右夹爪交接同一个刚性长端头，最后落入托台 |
| `cable_stretch` | `ibero/CableStretch-v0` | Core-0.1：两手各持一个端头，拉紧线束、保持张力、应对外力扰动 |

当前默认 demo 是 `cable_stretch`。它最直接体现了平台要发展的“材料—接触—双臂力控”能力。

不要把 `CableTension-v0` 与 `CableStretch-v0` 当作同义词：前者是旧的概念验证，后者才是本轮接触式双端张力任务。

### 1.3 本阶段没有在做什么

当前没有训练神经网络，没有语言模型控制机器人，没有 LeRobot 数据管线，也没有通用场景编辑器。运行 demo 时的动作由人为编写的脚本控制器产生，不是训练得到的策略。

当前材料数值仍为 `provisional`，即工程试验参数。项目已经可以检查某种等效模型是否实现正确，但还不能说它准确代表某一款产线线束。

<a id="part-2"></a>
## 2. 五分钟运行与观察

在项目根目录打开终端。如果使用已有开发环境：

```powershell
conda activate ibero-core
python -m ibero.demo --env cable_stretch --viewer --playback-rate 0.5
```

新环境的 Python 版本要求为 3.11 或以上；本地安装项目使用：

```powershell
python -m pip install -e ".[dev]"
```

`-e` 是可编辑安装：Python 导入本项目时指向 `src/ibero/`，修改源码后重新运行即可使用修改结果。项目使用根目录的 `scenes/` 与 `assets/`，目前应保留完整仓库结构，不要只复制一个 `ibero` 包目录。

### 2.1 应当看到什么

一个固定基座的 G1 上半身，左右腕各装一个 2F-85，两手夹着线束两端。初始线束有松弛，运行后两臂逐渐调整间距，让张力进入目标范围；中途有一个显式外力扰动，控制器随后恢复张力。

当前场景不是从桌面上寻找并抓起物体：初始抓取位置已经由菜谱安排好，reset 阶段让夹爪闭合形成初始接触。自动寻物与抓取规划不在本版本范围内。

| 操作 | 效果 |
|---|---|
| Enter | 用指定 seed 开始新一轮；结束后可再运行 |
| P | 暂停或继续 |
| N | 推进一个控制周期，默认 0.05 秒仿真时间 |
| R | 回放当前已经记录的轨迹 |
| B | 回看上一帧 |
| 关闭 MuJoCo 主窗口 | 结束审阅会话 |

初始窗口会等待，不会立刻跑完退出。`--playback-rate 0.5` 请求放慢播放；实际墙钟速度还受物理计算和图形渲染速度影响，不是硬实时保证。

### 2.2 不打开交互窗口也能运行

```powershell
python -m ibero.demo --env cable_stretch
python -m ibero.demo --env cable_handover
```

第一条看张力闭环，第二条看交接。需要保存产物时：

```powershell
python -m ibero.demo --env cable_stretch --save-trace artifacts/stretch_trace.npz --save-multiview artifacts/stretch_fourview.png --save-manifest artifacts/stretch_manifest.json
```

其中 PNG 仍需要可用的离屏渲染环境；纯数值运行不需要创建 viewer。四格面板是可选的 Tk 窗口，MuJoCo 主窗口本身仍需要图形桌面。

<a id="part-3"></a>
## 3. 项目目录地图

下面只展示理解平台所需的文件，省略缓存、网格文件清单及安装生成目录。

```text
IBERO/
├── pyproject.toml                    Python 包配置、依赖版本、命令入口
├── README.md                         快速启动与范围说明
├── IBERO__…建设方案.docx              平台总体目标，不是运行时配置
├── docs/
│   ├── 02_最小实现与协作说明.md       最小工作方式与角色划分
│   ├── 03_Core-0_实施契约.md          早期烟雾测试边界
│   ├── 04_Core-0.1_实施契约.md        决策、验收门槛、历史与补强记录
│   └── 04_Core-0.1_版本介绍.md        本文：当前实现的阅读地图
├── assets/vendor/
│   ├── unitree_g1/                   G1 的 MJCF、网格、许可证
│   └── robotiq_2f85/                 夹爪的 MJCF、网格、许可证
├── scenes/
│   ├── cable_handover/
│   │   ├── scene_config.yaml         机器人、材料、初始化、物理与工装
│   │   ├── constraints.yaml          安全阈值、夹持窗口与碰撞许可
│   │   ├── task_spec.py              交接任务的阶段与成功/失败判定
│   │   └── README.md                 菜谱使用方法与物理假设
│   └── cable_stretch/                同样四个文件，定义双端张力任务
├── src/ibero/
│   ├── __init__.py                   小型环境工厂：ibero.make()
│   ├── demo.py                       命令行主入口、选择环境与运行模式
│   ├── baselines.py                  接触式交接脚本
│   ├── review.py                     持久审阅循环与 Trace 轨迹
│   ├── multiview.py                  四视角渲染、信息面板、张力曲线
│   ├── check_scene.py                场景声明检查命令
│   ├── calibrate.py                  多 seed 保持、交接、腕部静载检查
│   ├── verify_core01.py              材料、力控与故障的综合验收
│   ├── verify_viewer.py              显式启动真实窗口的交互检查
│   ├── envs/
│   │   ├── cable_handover.py          Core-0.1 两个菜谱共用的运行环境
│   │   └── cable_tension.py           Core-0 旧环境
│   ├── core/
│   │   ├── scene_loader.py           读取、检查 YAML，计算场景标识
│   │   ├── scene_compiler.py         将合法菜谱交给模型构建器
│   │   ├── task_api.py               TaskState / TaskResult / TaskSpec
│   │   ├── sensors.py                腕部 F/T 的读数与顺序约定
│   │   ├── audit.py                  安全事件、峰值与损伤锁存
│   │   └── reproducibility.py        仿真源码哈希
│   ├── robots/
│   │   ├── g1_upperbody_2f85.py       Core-0.1 的 G1 裁剪、夹爪、工装组装
│   │   └── g1_upperbody.py           Core-0 构建器及共用资产路径/关节命名
│   ├── materials/
│   │   ├── cable.py                  材料参数、Flex/端头几何与连接
│   │   ├── mechanics.py              弯曲力、真实边张力与曲率观测
│   │   └── bench.py                  不依赖机器人脚本的材料试验台
│   └── control/
│       ├── resolved_rate.py          双臂末端增量 → 关节执行目标
│       └── tension.py                张力误差 → 双臂末端增量
├── tests/
│   ├── test_cable_tension.py         Core-0 回归
│   ├── test_core01_cable_handover.py  场景、机器人、接触交接与接口回归
│   └── test_physics_revision.py      材料、故障、力控、轨迹与安装检查
└── artifacts/                       生成的报告、PNG、NPZ、manifest
```

三条最有用的目录规则：

1. `assets/` 保存“原始模型长什么样”，`scenes/` 保存“本次实验如何配置”。两者不能混为一谈。
2. `src/ibero/` 实现运行机制，`tests/` 检查机制是否符合预期，`artifacts/` 保存检查证据。
3. P2、P3、P4 是职责划分，不是三个独立进程，也不是运行时的三个 AI agent。

各子目录的 `__init__.py` 主要用于 Python 包导出，不需要作为第一批重点阅读文件。

<a id="part-4"></a>
## 4. Core 究竟由哪些部分组成

### 4.1 “核心系统”不等于 `core/` 文件夹

完整的仿真核心横跨环境、机器人、材料、控制和公共基础设施。`core/` 只是存放装配协议、任务数据结构、传感和审计等公共部分。

| 部分 | 负责的问题 | 不负责的问题 |
|---|---|---|
| P4：`scenes/` | 用什么材料、初始构型、目标、安全约束；什么叫成功 | 不自行积分物理状态 |
| P2：`robots/` | 机器人和夹爪如何装配，工具轴在哪，哪些对象叫什么 | 不决定线束任务是否成功 |
| P3：`materials/` | 线束如何离散、连接、受力、弯曲和输出材料状态 | 不替机械臂制定动作策略 |
| `control/` 与 `baselines.py` | 怎么动双臂和夹爪 | 不通过直接改物体位姿制造成功 |
| `envs/` | 将上述部分组织成一次可运行的实验 | 不重新发明物理求解器 |
| `core/` | 校验、编译入口、任务状态、传感和安全审计 | 不是独立于其他目录的完整引擎 |
| MuJoCo | 动力学、接触、约束、状态积分与渲染能力 | 不知道 IBERO 的“任务成功”是什么意思 |

### 4.2 配置阶段与运行阶段是两张不同的关系图

配置阶段是“构造世界”：

```text
scene_config.yaml + constraints.yaml
                  ↓
             SceneLoader
                  ↓ ValidatedScene
             SceneCompiler
                  ↓
      build_g1_handover_model()
       ├── G1/2F-85 资产 → 机器人与工具
       ├── cable.py     → Flex、端头与连接
       └── 工装配置     → 地面、接收台、目标标记
                  ↓ compile()
    CompiledScene(model, handles, cable_parameters)

task_spec.py → 动态加载 TASK_SPEC → 环境中的任务裁判
```

运行阶段是“推进世界”：

```text
脚本策略 → env.step(action) → 末端/关节控制
                  ↓
        材料弯曲力 + MuJoCo 物理积分
                  ↓
       材料读数 / 实际接触 / 腕部传感器
                  ↓
          SafetyAuditor + TaskSpec
                  ↓
      observation、reward、结束标志、info
```

理解这两张图，就不容易把“生成模型的代码”与“每一帧运行的代码”混在一起。

### 4.3 当前架构有意保持具体

`SceneCompiler` 目前直接调用 G1 专用的 `build_g1_handover_model()`。这个构建器还组装工装并调用 P3 材料构建，不是纯机器人资产加载器。

因此：项目已经有清楚的职责分界，但还没有完成多机器人、多引擎、任意物体的插件化架构。保留这条具体链路，是为了先把一种双臂线束操作做可信，而不是先堆叠抽象接口。

<a id="part-5"></a>
## 5. 一条命令如何找到场景并启动

以 `python -m ibero.demo --env cable_stretch --viewer` 为例。

### 5.1 第一层：Python 找到主程序

[pyproject.toml](../pyproject.toml)指定 Python 包位于 `src/` 下。安装后，`ibero.demo` 对应 [src/ibero/demo.py](../src/ibero/demo.py)。

这个项目不是用一个根目录的 `main.py` 启动。`demo.py` 的 `main()` 解析命令行，文件末尾的 `if __name__ == "__main__"` 调用它。安装产生的 `ibero-demo` 命令也指向同一个 `main()`。

### 5.2 第二层：命令名称变成环境对象

`demo.main()` 把 `cable_stretch` 映射为 `ibero/CableStretch-v0`，再调用 [ibero.make()](../src/ibero/__init__.py)。

`ibero.make()` 是项目自己写的小型工厂函数，不是自动发现所有 `scenes/` 目录的插件注册系统，也不是已经注册到 `gym.make()` 的全局环境表。

最容易困惑的一点：

```text
CableHandover-v0 ──→ CableHandoverEnv(scene_path=cable_handover)
CableStretch-v0  ──→ CableHandoverEnv(scene_path=cable_stretch)
```

两个 Core-0.1 名字对应同一个 Python 类。类名保留了历史命名，实际已经服务于两个线束菜谱。

### 5.3 第三层：环境找到具体文件

[envs/cable_handover.py](../src/ibero/envs/cable_handover.py)中的 `DEFAULT_SCENE_PATH`，根据该源码文件的绝对位置向上定位项目根目录，再拼接 `scenes/cable_handover`。

`CableStretch-v0` 在工厂中把默认路径换成同级的 `cable_stretch`。因此从其他工作目录执行已安装的包时，默认场景不依赖终端当前目录；但 `--scene` 和输出文件的相对路径仍按当前终端目录解释。

环境构造函数接着完成：

1. `SceneLoader.validate(scene_path)`：读取两份 YAML，检查字段和值，确认任务文件存在。
2. `_load_task_spec()`：用 `importlib` 加载该目录下的 `task_spec.py`，取得 `TASK_SPEC` 实例。
3. `SceneCompiler.compile()`：得到模型、对象索引和材料参数。
4. 创建 `MjData`、`CableMechanics`、低层控制器、审计器以及动作/观测空间。

`materials.cable.end_layout` 决定 `env.dual_end`。当前 demo 根据这个布尔值选择张力脚本或交接脚本，不是只根据文件夹名字选择控制器。

### 5.4 第四层：选择运行循环

无 `--viewer` 时：`demo.rollout()` → `env.reset()` → `_run_episode()` → 重复策略与 `env.step()`。

有 `--viewer` 时：`demo.rollout()` → [review.run_review()](../src/ibero/review.py) → 持久窗口与按键事件循环。这里等待 Enter，并负责暂停、单步和回放。

主程序是“组织调用的人”，真正推进物理状态的是环境中的 `step()`。

<a id="part-6"></a>
## 6. 引擎、模型与动态状态

### 6.1 当前技术栈

以 [pyproject.toml](../pyproject.toml)中的锁定版本为准：

| 组件 | 本项目用途 |
|---|---|
| MuJoCo 3.8.1 | 物理引擎与渲染基础；机器人、接触、Flex 和约束在这里求解 |
| Gymnasium 1.2.3 | 约定 `reset()`、`step()`、动作空间和观测空间；不是物理引擎 |
| NumPy | 坐标运算、张力/弯曲计算、控制器线性代数 |
| PyYAML | 将场景 YAML 读成 Python 数据 |
| Pillow / Tkinter | 图像输出与可选的四视角桌面面板 |
| pytest | 自动化回归测试 |

Python 负责模型组织、控制逻辑和材料补充计算；MuJoCo 的本地引擎负责核心物理运算。项目没有自己从零编写一个完整物理引擎，也没有在本版本使用 Isaac Sim、ROS2 或游戏引擎。

### 6.2 必须区分的三个 MuJoCo 对象

| 名字 | 可以怎样理解 | 在本项目中何时出现 |
|---|---|---|
| `MjSpec` | 可编辑的模型草稿：有哪些 body、joint、geom、传感器 | P2/P3 读取与拼接资产时 |
| `MjModel` | 编译后的模型描述：拓扑、质量、参数、索引、执行器等 | `spec.compile()` 之后 |
| `MjData` | 某一次仿真的动态状态：位置、速度、控制输入、接触、传感读数等 | 环境创建，并在 reset/step 中变化 |

简化记忆：Spec 是“装配草稿”，Model 是“编译后的机器结构和参数”，Data 是“这台机器此刻的状态”。

`MjModel` 并非绝对不可写。例如当前目标位置随机化会修改工装 body 的模型位置；但每一帧机器人运动主要体现在 `MjData`，不是反复生成整份模型。

### 6.3 MJCF、网格与物理几何

资产目录的 XML 使用 MuJoCo 模型描述格式 MJCF。网格文件主要描述外形，MJCF 同时声明刚体关系、质量惯性、关节、执行器和接触几何。

“画出来的形状”与“参与接触的形状”不一定完全相同。比如目标区绿色标记被设置为不碰撞；真正支撑端头的是另一个名为 `receiver_support` 的 box geom。仅仅看见一个绿色区域，并不说明这里有承重的物理台面。

### 6.4 `mj_forward()` 与 `mj_step()` 不一样

`mj_forward()` 根据已有状态更新几何、动力学和传感相关计算，不推进仿真时间；`mj_step()` 推进一个物理时间步。

因此 reset 后或回放恢复状态后会用 `mj_forward()` 刷新画面所依赖的位置；正常运动靠 `mj_step()` 积分，而不是不断修改物体坐标。

<a id="part-7"></a>
## 7. 名称、ID、数组地址：如何索引仿真对象

这是读懂项目最关键的技术环节之一。

### 7.1 先区分对象类型

| 对象 | 含义 | 项目中的例子 |
|---|---|---|
| body | 刚体及其局部坐标系 | `cable_terminal`、`left_wrist_yaw_link` |
| joint | 刚体之间允许怎样运动 | 肩关节、端头的平移关节和球关节 |
| geom | 外形/碰撞几何 | `cable_terminal_geom`、指垫 geom |
| site | 标记点与坐标系，本身不是抓取约束 | `left_pinch`、`left_wrist_ft`、`target_region` |
| actuator | 引擎中的驱动通道 | 双臂关节执行器、`left_fingers_actuator` |
| sensor | 引擎提供的测量通道 | `left_wrist_ft_force` |
| flex | 柔性节点与元素集合 | `cable_flex` |
| equality/connect | 材料连接等需要求解的约束 | `terminal_to_flex` |

TCP 是工具中心点，本项目用夹爪的 `pinch` site 作为末端控制参考。`site` 可以用于算距离、雅可比或传感器坐标，但放一个 site 不会把物体吸附到这里。

### 7.2 字符串名字如何变成运行时索引

模型编译后，引擎主要按整数索引访问对象：

```python
terminal_id = env.model.body("cable_terminal").id
terminal_position = env.data.xpos[terminal_id]

pinch_id = env.handles.left_pinch_site_id
pinch_position = env.data.site_xpos[pinch_id]
pinch_rotation = env.data.site_xmat[pinch_id].reshape(3, 3)
```

以上代码说明的是索引方式。`xpos` 是 body 位置，`site_xpos` 是 site 位置，不可用 site ID 去索引 body 的数组。

P2 构建器把常用索引集中保存为 `HandoverModelHandles`；P3 使用 `CableHandles` 保存端头、夹持标记、Flex 等索引。上层不需要到处硬编码“第几个 body 是左手”。

注意：修改模型组成后，整数 ID 可能改变。应依赖稳定名字重新查找，而不是把某一轮编译产生的 ID 写进场景 YAML。

### 7.3 ID 也不等于状态数组的地址

读取关节角时不能想当然写 `data.qpos[joint_id]`。不同关节在状态向量中占用的长度可能不同。

当前代码的实际映射是：

```text
arm_actuator_ids
    ↓ model.actuator_trnid[..., 0]
arm_joint_ids
    ├── model.jnt_qposadr[...] → data.qpos 的位置地址
    └── model.jnt_dofadr[...]  → data.qvel 的速度地址
```

传感器也是类似的两步关系：

```text
sensor_id
    ↓ model.sensor_adr / model.sensor_dim
data.sensordata[address : address + dimension]
```

这正是 [core/sensors.py](../src/ibero/core/sensors.py)存在的意义之一：统一解析，避免读错缓冲区。

### 7.4 Flex 有自己的索引层

[CableMechanics](../src/ibero/materials/mechanics.py)先找到 `cable_flex` 的 ID，再通过 `flex_edgeadr/edgenum` 与 `flex_vertadr/vertnum` 找出属于这条线束的边和节点切片。

随后使用 `flex_vertbodyid` 找到节点对应的 body，通过关节自由度地址找到需要施加弯曲力的位置。这条索引链把“材料节点的力”送到正确的引擎自由度，而不是送到机械臂关节上。

### 7.5 两个历史名字不要误读

`HandoverModelHandles.left_anchor_site_id` 与 `right_anchor_site_id` 是低层控制器保留的兼容字段。在 Core-0.1 中，它们指向真实夹爪的 `pinch`，不是把线束锚在手腕上的约束。

`cable_axial_tendon` 也是历史名字。当前它的刚度和阻尼为零，只保留距离测量用途，不是线束拉力的来源。

<a id="part-8"></a>
## 8. P2 机械师：G1 和夹爪怎么组装

入口是 [build_g1_handover_model()](../src/ibero/robots/g1_upperbody_2f85.py)。其输入为场景配置和约束，输出为 `model`、`handles`、`CableParameters`。

### 8.1 原始模型并不直接等于最终工作台

构建过程主要包括：

1. 从 `assets/vendor/unitree_g1/g1.xml` 读取 G1。
2. 删除浮动根关节，使实验使用固定基座。
3. 删除左右腿部子树和不再适用的原始关键帧。
4. 保留上半身，运行时维持非双臂关节的准备姿态。
5. 添加工装、目标位置、相机和光照。
6. 根据末端类型处理原生手外观并装配两个 2F-85。
7. 添加腕部 F/T 接口，调用 P3 加入线束，统一编译并解析索引。

修改发生在运行时的 `MjSpec`，不是直接重写 vendor XML。

共用的 `_spec_from_snapshot()` 位于 [g1_upperbody.py](../src/ibero/robots/g1_upperbody.py)：它从文本与资源字节构建模型，避免直接让原生文件加载器处理本项目的中文路径。这也是为什么 Core-0.1 仍导入这个看似“旧版本”的文件。

### 8.2 夹爪的坐标安装

场景中使用：

```yaml
robot:
  preset: g1_upperbody_v0
  end_effector:
    type: robotiq_2f85
```

G1 腕部前向与 Robotiq 资产的前向定义不同。当前通过绕腕部 Y 轴旋转 90°，把夹爪工具 +Z 对准腕部 +X；转接点还具有 0.065 m 的腕部局部 X 偏移。

这里要区分两个问题：安装变换描述“夹爪怎样固定在手腕上”；`robot.arm_qpos` 描述“机械臂怎样摆放”。不能依靠扭转手腕来掩盖错误的安装坐标。

两份夹爪资产通过 `left_`、`right_` 前缀附加，避免左右部件重名。比如夹爪源模型自身就有左右手指，所以 `left_right_pad1` 表示“机器人左手夹爪的右侧指垫”，不是左右臂之间的某个部件。

### 8.3 14 个机械臂关节的顺序

`ARM_JOINTS` 定义先左臂、后右臂；每臂依次为：

```text
shoulder_pitch → shoulder_roll → shoulder_yaw → elbow
               → wrist_roll → wrist_pitch → wrist_yaw
```

`robot.arm_qpos` 按这个顺序填 14 个弧度值。reset 会检查是否超出模型关节范围。

### 8.4 “原生手”选项的真实含义

当前资产中的 `g1_native_hand` 只有固定手部外观，不是完整的可驱动灵巧手。构建器可以保留这种外观用于模型检查，但 Core-0.1 任务环境明确拒绝以此运行。

默认 2F-85 模式先去掉两个原生手 visual geom，再挂夹爪，因此不会把两套手叠在一起。未来接入真实灵巧手，需要补关节、碰撞、驱动、工具坐标和抓取验证，不是改一个字符串就完成。

<a id="part-9"></a>
## 9. P3 材料专家：线束究竟怎么模拟

这是本项目最关键的部分。可以把 P3 分成三个层次：参数定义、可求解的物理表示、运行中的材料响应与测量。

```text
scene_config.yaml / materials.cable
                  ↓ CableParameters.from_scene()
               cable.py
      生成动态端头、Flex 节点、边与连接
                  ↓ 编译进同一个 MuJoCo 模型
             mechanics.py
      每子步施加弯曲力；读取边张力、弧长、曲率
                  ↓
     控制器使用张力；审计器使用阈值；任务使用材料状态
```

### 9.1 参数由场景提供，不藏在 demo 里

从 [cable_stretch/scene_config.yaml](../scenes/cable_stretch/scene_config.yaml)读取：

| 字段 | 物理含义 | 当前实现中的作用 |
|---|---|---|
| `length_m` | 柔性段原始弧长 | 初始化形状、线束总质量、弯曲离散尺度 |
| `outer_diameter_m` | 外径 | Flex 接触半径为外径的一半 |
| `linear_density_kg_m` | 单位长度质量 | 总质量 = 线密度 × 原长 |
| `terminal` | 刚性端头尺寸与质量 | 动态端头 geom 与质量参数 |
| `axial_stiffness_n_m` | 全长等效轴向刚度 k | 决定拉长后产生多大的轴向恢复力 |
| `damping` | 全长等效轴向阻尼 c，单位 N·s/m | 决定伸缩速度带来的阻尼力 |
| `bending_stiffness_nm2` | 等效弯曲刚度 EI | 决定中心线弯曲的能量代价 |
| `minimum_bend_radius_m` | 最小允许弯曲半径 | 曲率报警阈值，不是几何硬限位 |
| `friction` | 材料接触摩擦参数 | 写入端头/Flex 的接触定义 |
| `segments` | 柔性段的离散边数 N | 默认 12 段，对应 13 个节点 |
| `end_layout` | `handover` 或 `dual_end` | 长端头加尾配重，或两个同规格端头 |
| `initial_span_m` | 端头中心初始距离 | 与弧长共同决定初始松弛形状 |
| `parameter_source` | 参数出处 | 区分 provisional、measured、datasheet 等来源标记 |

尺寸、质量、最小弯曲半径不足以唯一决定真实线束的力学行为；还需要轴向刚度、弯曲刚度、阻尼和接触数据。当前代码不会根据外径自动识别内部铜丝、绝缘层或等效弹性模量。

### 9.2 为什么既有长度，又有中心距

默认双端任务中：柔性段弧长为 0.26 m，端头中心距为 0.30438 m，两个端头各长 0.08 m。

连接点在两个端头相向的边缘，因此柔性段两端的直线间距约为：

```text
0.30438 - 0.08/2 - 0.08/2 = 0.22438 m
```

0.26 m 的线束放在相距 0.22438 m 的连接点之间，自然有松弛。不能因为中心距大于 0.26 m，就断言线束初始化已经拉长。

`_flex_subspec()` 用一条正弦形中心线描述松弛，通过二分调节弧幅，让各离散边长之和等于指定原长。初始化曲线不意味着线束具有这个永久自然弯曲形状；后续仍会按材料力、重力与接触变化。

### 9.3 Flex 在这里是什么

[materials/cable.py](../src/ibero/materials/cable.py)生成 `flexcomp type="direct" dim="1"`：代码明确给出节点坐标和相邻节点的连接关系。

可以直观理解为一串有质量、能移动的节点，以及连接它们的柔性线段。接触外形表现为沿线段展开的胶囊状元素，不是只有一条无厚度的绘图线。

节点质量来自整条线束质量；端头另外拥有自己的质量。端头通过三个平移关节和一个球关节获得平移、转动自由度，不是固定在世界中的装饰物。

当前只有一种已验收的表示：`representation: flex`。旧 `capsule_chain` 曾只是同一实现的另一标签，现在已被拒绝。不要将“Flex 的接触形状像胶囊链”误解为“项目已有另一套独立刚体链后端”。

### 9.4 端头和线束如何连接

两处 `connect` 约束将线束端点连接到端头边缘。它们表达材料本来就与端头相连，而不是模拟手对物体的抓取。

```text
夹爪指垫 ← 接触与摩擦 → 刚性端头 ← 材料 connect → Flex 节点与边
```

端头—线束连接由引擎的约束求解维持；夹爪—端头之间没有人为加上的抓取 weld。松开夹爪后，端头仍与线束相连，但可以离开夹爪。

当前 connect 并不描述完整的端头抗扭/抗弯连接本构；它保证连接位置关系，不能据此宣称已经模拟真实接插件的全部机械特性。

### 9.5 轴向刚度与阻尼怎么进入引擎

场景给的是整条线束的等效刚度 k。将它离散为 N 个串联段时，各段使用 `N*k`；阻尼类似使用 `N*c`，避免仅增加分段数就把整条线束错误地变软。

每段的轴向力按以下形式理解：

```text
第 i 段有符号轴向力
    = 段刚度 ×（当前边长 - 该边原长）
    + 段阻尼 × 当前边长变化速度
```

边的实际刚度与阻尼写入 MuJoCo Flex；引擎在物理积分中施加这部分力。`mechanics.py` 使用相同的力律从引擎真实边长和速度恢复读数，不再用两个端头的直线距离虚构整条线束拉力。

当前 `cable_tension_n` 报告所有边中最大的正轴向力，另报首尾两段的正张力。动态扰动时，各段张力不一定相等，所以“最大值”“两端值”和“平均值”不是一回事。

截取正值是张力观测的定义，不表示引擎把所有轴向压缩力都删除。当前模型也不是一个完整的只受拉连续体理论。

### 9.6 弯曲不是只填一个 YAML 字段

线束只具有轴向弹簧还不够：它可以改变方向，却没有我们需要的受控弯曲刚度。

[bending_energy_force()](../src/ibero/materials/mechanics.py)根据相邻线段方向变化定义能量：

```text
E = EI / (2h) × Σ |t[i+1] - t[i]|²
```

其中 EI 是场景提供的弯曲刚度，h 是采用的平均离散步长，t 是各段单位方向。相邻段方向越不一致，弯曲能量越大；EI 越大，就越不容易弯曲。

代码计算这个能量对节点位置的负梯度，得到各节点的弯曲恢复力，再写入这些节点对应的 `data.qfrc_applied` 自由度。

因此目前材料实现是：**MuJoCo 提供柔性节点、接触、连接约束与轴向响应，IBERO 在每个子步补充中心线弯曲力。** 不是把一个“完整真实线束材料”名称交给 MuJoCo，它就自动知道所有材料性质。

### 9.7 为什么每个物理子步都要更新弯曲力

`CableMechanics.apply()` 先通过 `mj_kinematics()` 和 `mj_flex()` 根据当前 `qpos` 更新节点几何，再计算弯曲力。

原因是积分后的某些派生位置缓冲可能仍对应此前的计算时刻。若一直用滞后的位置施加材料力，可能给振动注入非物理能量。该更新顺序属于材料与引擎结合的关键细节，不只是性能优化。

### 9.8 曲率、报警与损伤是不同事情

`state()` 根据相邻线段方向与长度估计曲率，再转换成最小局部弯曲半径。当它小于配置值时，输出 `bend_alarm`。

当前行为是报警与任务条件检查，不是强行把线束几何裁剪到最小半径，也不会自动断裂。超拉损伤另由审计器按张力和持续时间判断。

### 9.9 哪些材料能力还没有

没有扭转本构，没有线束自接触，没有内部多股结构、截面应力、绝缘层破坏或物理断裂。当前碰撞设置也会屏蔽 Flex 与自身端头的装配接触，避免连接处互相顶开；它不是所有物体之间都无条件碰撞。

真实材料标定需要独立的拉伸、弯曲/悬垂、摩擦等实验。把来源字段从 `provisional` 改成 `measured` 只是改变声明，不能替代真实测量。

<a id="part-10"></a>
## 10. 接触、抓取和三种不同的力

### 10.1 “夹爪夹住端头”在物理上如何发生

脚本给夹爪闭合指令，夹爪执行器驱动手指运动；指垫与端头的几何接触产生约束与接触力，摩擦使端头能够随夹爪运动。与此同时，线束对端头的拉力又通过接触传回夹爪。

接触与材料力在同一 MuJoCo 模型中耦合求解，不是先把物体跟随手移动，再单独画一条弯曲线。

但“引擎发现某个接触”不等于“任务确认抓住了”。这是下面一层语义判断。

### 10.2 环境如何确认抓取

在 [CableHandoverEnv](../src/ibero/envs/cable_handover.py)中，依次看四个方法：

1. `_terminal_pad_contact()`：检查实际 `data.contact`，要求相对的两组指垫都接触到本侧应该抓取的端头。
2. `_grasped_raw()`：再检查夹持标记到 pinch 的空间距离和夹爪开度。
3. `_update_grasp_history()`：将端头中心变换到夹爪局部坐标，记录它相对夹爪的位置。
4. `_grasped()`：要求最近三次控制采样均满足基础条件，且相对位置变化小于 4 mm。

这样会排除“只有一个手指蹭到”“空夹爪闭合”“物体在两指之间快速滑走”等情况。

三次采样之间默认各隔 0.05 s，首末间隔约 0.10 s。不能将三帧简单当成一个经过完整积分的 0.15 s 保持时长。这仍是工程抓取判据，不是对任意物体的完整力闭合证明。

### 10.3 必须区分三类力

| 力 | 来自哪里 | 用来回答什么问题 |
|---|---|---|
| 指垫/端头接触力 | 引擎求解的接触反作用；可视化显示接触力箭头 | 两个物体如何通过接触相互作用 |
| 腕部 F/T | wrist site 上的 force、torque 传感器 | 手腕测到怎样的合力与合力矩 |
| 线束张力 | Flex 各边的实际轴向弹簧/阻尼响应 | 线束本身哪里被拉得最紧 |

腕部传感器可能同时受到工具、端头重力、加速度、线束载荷等影响。它不是把接触点力直接复制过来，也不能直接当作线束最大内部张力。

`core/sensors.py` 将腕部读数整理为 `[Fx, Fy, Fz, Tx, Ty, Tz]`，坐标系为对应 wrist site 的局部坐标，不是默认世界坐标。接触箭头则是可视化提示，不能通过箭头像素长度替代数值读数。

### 10.4 当前“力控操作”的准确含义

本版本实现的是：根据测量张力调整双臂运动，让接触夹持的线束进入指定受力范围。

它不是直接发送“关节力矩 = 某值”，也不是让夹爪精确跟踪一个法向夹紧力目标。当前夹爪接收的是开闭驱动命令；双臂张力外环最终仍通过关节位置执行器实现。

所以当前已具备“接触式物体操作 + 张力反馈闭环”，尚未具备“任意接触面的法向力控制、精密轻放或插接”。

<a id="part-11"></a>
## 11. 双臂如何执行张力控制

可以把控制分成两层，而不是把所有控制行为都叫作一个 controller。

### 11.1 外层：想达到什么张力

[control/tension.py](../src/ibero/control/tension.py)中的 `CableStretchScript.action(env)` 读取当前张力，输出 14 维归一化动作。

主要步骤：

1. 对张力做一阶低通滤波，减轻高频抖动对运动命令的影响。
2. 根据“目标张力减测量张力”计算期望的两端分离速度。
3. 限制速度，避免一次动作过大。
4. 左右臂各承担一半相对位移，同时保持双手共同中心与初始工具姿态。
5. 若原始张力过高，进入卸载分支，让两端靠近。

普通张力调节可以用一句公式理解：

```text
分离速度 = 限幅（导纳增益 × 张力误差）
```

力偏小就向外拉，力偏大就减小间距。这是简化的导纳思想：用力反馈生成运动修正，不是完整的多自由度质量—阻尼—刚度导纳系统。

当前普通分支使用滤波张力；过载判断使用原始张力，并带有回差：超过 5 N 卸载，下降到 3 N 以下才退出，以免在同一个阈值附近频繁切换。

默认目标为 2 N，容差为 ±0.4 N。达到 1.68 N 可以处于合格范围，不意味着控制器精确达到 2.000 N。

### 11.2 内层：末端增量如何变成关节运动

[control/resolved_rate.py](../src/ibero/control/resolved_rate.py)中的 `BimanualResolvedRateController` 做以下事情：

```text
归一化末端动作
    ↓ 乘动作尺度
末端平移/转动增量
    ↓ 除控制周期
末端速度要求
    ↓ site 雅可比 + 阻尼最小二乘
双臂关节速度
    ↓ 限速、积分、关节范围裁剪
关节目标位置
    ↓ 写入 data.ctrl
MuJoCo 关节执行器使机器人运动
```

雅可比描述“各个关节微小运动会让工具怎样移动”；阻尼最小二乘用于在奇异或接近奇异姿态时抑制过大的关节速度。当前采用数值阻尼项 `2.5e-3`，关节速度限制为 ±2 rad/s。

目标关节位置是随动作累积更新的执行目标，不是直接给 `data.qpos` 赋值。因此即使目标在移动，实际运动也仍受驱动器、负载、接触和关节范围影响。

这不是运动规划器。它不会自动找绕障路径，也不保证每一个要求的末端姿态都可达。

### 11.3 14 维动作不等于 14 个关节角

动作排列如下：

```text
[左 dx,dy,dz,drx,dry,drz,grip,
 右 dx,dy,dz,drx,dry,drz,grip]
```

每个值归一化到 `[-1,1]`。平移尺度为 0.012 m/控制周期，转动尺度为 0.10 rad/控制周期；增量按世界坐标表达。夹爪 `grip=-1` 为张开，`+1` 为闭合，映射到模型驱动通道的 `[0,255]`，不是 255 N。

两臂机械结构恰好也是 14 个关节，但动作的 14 维来自“两组 6D 末端增量 + 两个夹爪命令”，不能混用。

### 11.4 交接为什么是另一个脚本

[baselines.py](../src/ibero/baselines.py)中的 `CableHandoverScript` 按接近、闭合、左释放、移向目标、右张开/退出等阶段产生动作。它使用同一个低层控制器，但不像双端任务那样主要围绕张力误差调节距离。

脚本负责尝试完成任务，任务文件负责独立评价结果。这两者分开，才能在脚本没有成功时如实记录失败。

当前脚本可以直接读取 `env.data`、工具旋转与抓取状态，属于使用仿真完整状态的工程基线，不是只依赖摄像头观测的已部署策略，也不是已经冻结的未来学习策略接口。

<a id="part-12"></a>
## 12. 一个 reset 和一个 step 内部发生什么

### 12.1 `reset(seed)`：安排可重复的起点

reset 不重新读取并编译全部资产，而是在已编译模型上建立新一轮状态：

1. 初始化随机数并清除回放状态、抓取历史。
2. `mj_resetData()` 重置动态状态。
3. 按 YAML 给定范围采样目标位置和扰动力幅度。
4. 设置双臂准备关节角，重置控制器目标。
5. 交接任务左闭右开；双端任务两侧闭合。
6. 暂时关闭重力，积分约 1.2 s，让指垫形成接触与初始抓取。
7. 恢复重力、仿真时钟归零，清除上一轮任务/审计状态。
8. 检查初始禁止碰撞，返回首个 `obs, info`。

关闭重力只发生在这段初始化沉降过程中，正式回合恢复重力。它是一个明确的实验初始化便利，不是对现实中从任意状态抓取过程的完整模拟；初始化沉降时间也不包含在正式回合的计时和损伤历史中。

### 12.2 `step(action)`：一次动作不等于一个物理步

当前两个 Core-0.1 默认菜谱都设置：

```text
物理步长：0.0002 s，即 5000 Hz
控制周期：0.05 s，即 20 Hz
一次 env.step：250 个物理子步
```

环境先执行一次 `controller.apply()`，再在一个控制周期内重复：

```text
更新当前 Flex 几何
    ↓
计算并写入节点弯曲力
    ↓
按实验时间决定是否施加扰动外力
    ↓
mj_step()：接触、材料与机器人一起推进
    ↓
数值异常检查、张力与禁止碰撞审计
    ↓
进入下一个物理子步
```

250 个子步结束后，环境刷新派生状态、更新抓取历史、记录弯曲/张力报警，再构造 `TaskState` 交给裁判。

这解释了为何安全检查不能只做 20 次/秒：短暂的超拉和接触可能在两次控制更新之间出现。当前损伤与禁止碰撞按物理子步审计；抓取历史、任务保持时间、曲率/控制报警按控制采样更新，频率并不完全相同。

### 12.3 三种“力输入”写到哪里

| 输入 | 写入位置 | 实际意义 |
|---|---|---|
| 机器人/夹爪执行命令 | `data.ctrl` | 让模型执行器驱动关节 |
| P3 弯曲恢复力 | 节点对应的 `data.qfrc_applied` | 对材料自由度施加自定义本构力 |
| 可重复实验扰动 | 中间节点 body 的 `data.xfrc_applied` | 施加指定世界系外力 |

扰动不是偷偷移动线束位置。默认在 4.0–4.4 s 施加向下约 0.6 N，seed 控制 ±10% 的幅度变化，实际取值进入 manifest。

<a id="part-13"></a>
## 13. P4 菜谱作者与任务裁判

### 13.1 为什么要拆成三个主要文件

“`scene_config.yaml + task_spec.py`”是最小贡献思路。现在为了让安全限制容易审阅，又独立出了 `constraints.yaml`。

| 文件 | 应当放什么 | 示例 |
|---|---|---|
| `scene_config.yaml` | 实验世界与可调实验设置 | 材料、臂姿态、步长、初始中心距、张力目标 |
| `constraints.yaml` | 安全与几何判据 | 8 N/0.04 s 超拉阈值、夹持窗口、碰撞许可 |
| `task_spec.py` | 与时间、阶段有关的任务语义 | 扰动后连续合格 2 s 才成功 |
| `README.md` | 作者对场景的解释 | 参数来源、运行方法、已知简化 |

场景文件没有取代程序：它们为已实现的机制提供参数。增加一个 YAML 字段，只有在 loader 允许且相关构建器/控制器实际消费它时才会有效。

### 13.2 loader 与 compiler 分别做什么

[SceneLoader](../src/ibero/core/scene_loader.py)负责读取和声明检查，包括已支持字段、材料表示、数值范围、来源标记、控制/物理周期关系、材料保守步长与阈值关系等。

[SceneCompiler](../src/ibero/core/scene_compiler.py)负责把已验证声明交给当前模型构建器，产生可运行的 MuJoCo 模型。

`python -m ibero.check_scene scenes/cable_stretch` 执行的是声明检查，不会完成机器人可达性、接触稳定性与任务成功率验证。配置能通过检查，不等于实验一定成功。

### 13.3 任务只拿“实验事实”，不直接操作物理引擎

[core/task_api.py](../src/ibero/core/task_api.py)定义三个主要概念：

- `TaskState`：环境整理出的物体/工具位置、夹爪开度、抓取标志、张力、步数与补充状态。
- `AuditSnapshot`：审计器当前的损伤、禁止碰撞、峰值与事件信息，定义在 `audit.py`。
- `TaskResult`：裁判给出的奖励、阶段、成功/失败、结束标志与指标。

`TaskSpec` 是任务必须继承的基类，核心方法为 `reset()` 和 `evaluate(state, audit)`。每份任务文件最后暴露 `TASK_SPEC` 实例，环境动态加载它。

注意当前接口的实际成熟度：`TaskState.contacts` 仍为空集合，接触语义主要通过 `grasps` 与 `extra` 中的 `left_contact/right_contact/supported` 提供；不要假设已有通用完整接触图。冻结的数据类也不是深度不可变的安全容器，任务文件应由可信作者提供；文本检查并不是恶意 Python 的沙箱。

### 13.4 双端张力任务怎样判成功

[cable_stretch/task_spec.py](../scenes/cable_stretch/task_spec.py)要求：

1. 左右两端保持抓取。
2. 当前原始张力误差不超过容差。
3. 当前没有弯曲半径报警。
4. 实验扰动已经结束。
5. 上述条件连续保持指定时间，默认 2 s。

中途不满足就清零保持计时。它不会因某一帧达到目标、滤波曲线好看，或脚本自称完成而判成功。

失去任意一端所需的接触连续达到四次控制采样、且此前形成过双端保持，会判为 `dropped`；并不是一定要两只手同时完全松开才失败。

### 13.5 交接任务怎样判成功

[cable_handover/task_spec.py](../scenes/cable_handover/task_spec.py)记忆是否出现过左持有、双持有、左释放，再检查最终端头位置和支撑。

当前脚本在托台上方约 0.10 m 释放并退出，端头靠重力落入真实托台。双手不再保持抓取、右夹爪已张开、端头在 4 cm 目标范围内、有支撑且低速持续 0.5 s 才成功。

任务的阶段与脚本的阶段不需要一一同名。例如脚本可以叫 `deposit`，任务可能仍在等待最终支撑成立。面板中的 `stage` 与 `controller_phase` 是两个不同来源。

### 13.6 审计器不是任务脚本自己的成功开关

[SafetyAuditor](../src/ibero/core/audit.py)记录张力峰值、连续超限时间和禁止碰撞。一旦达到损伤判据，标志锁存，本回合后续卸载也不会清除。

默认张力任务超过 8 N 连续 0.04 s 判损伤。因此“峰值超过 8 N”与“已触发损伤”不能画等号：还要看持续时间。这种区分同时保留在事件和峰值记录中。

碰撞审计覆盖跨臂、同臂非邻接部件、臂与躯干、机器人与工装等实际接触；会排除同一夹爪机构、近邻装配和材料操作接触，并支持声明的 body 对许可。这是当前工程碰撞规则，不是对所有可能危险行为的完整安全证明。

`self_collision_flags.any_forbidden` 表示这类禁止碰撞；兼容键 `left_right` 当前也表达同一个总标志，不能据此认定事件一定发生在左右臂之间，应看记录的 body 对。

<a id="part-14"></a>
## 14. 观测、动作和运行结果

### 14.1 Gymnasium 形式的最小使用方式

```python
import ibero
from ibero.control.tension import CableStretchScript

env = ibero.make("ibero/CableStretch-v0")
try:
    obs, info = env.reset(seed=7)
    policy = CableStretchScript()
    for _ in range(env.max_episode_steps):
        action = policy.action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(info["is_success"], info["failure_reason"])
finally:
    env.close()
```

这是当前工程脚本的实际调用模式，不代表未来学习策略必须接收整个 `env`。

`terminated` 表示本轮因成功或任务失败等自然终止；`truncated` 表示时间/步数限制导致截断。不能把“终止”直接理解为“成功”。

### 14.2 当前四组观测

| 字段 | 维数 | 内容 |
|---|---|---|
| `proprio` | 30 | 14 个机械臂关节位置、14 个关节速度、两侧夹爪开度 |
| `ee_pose` | 14 | 每侧 3D pinch 位置、3D 工具 +Z 方向、1 个开度 |
| `wrench` | 12 | 左右腕各 6 维 F/T，均在对应 site 局部坐标 |
| `cable` | 3 | 最大正边张力、当前柔性段弧长、主端头到目标的距离 |

`ee_pose` 这个名字并不表示已经提供完整的两个 SE(3) 位姿：一个工具方向向量不能唯一确定绕该轴的旋转。脚本需要完整旋转时会读取 `site_xmat`。

当前观测没有直接包含摄像头图像；画面由单独渲染接口产生。张力是可从模拟器内部访问的材料状态，也没有在这里自动加上传感噪声、延迟或估计误差。

### 14.3 `info` 应当怎样读

最常用字段为：

```text
stage                 任务裁判判断的当前阶段
is_success            是否真正满足成功条件
failure_reason        cable_damage / self_collision / dropped / timeout 等
metrics               任务指标，如保持时间、张力误差、目标距离
audit.material        张力、两端张力、弧长、曲率/弯曲报警等
audit.events          带时间戳的安全/报警事件
controller_phase      demo 额外添加的脚本阶段，并非 env.step 固有字段
```

排查失败时先看 `failure_reason` 和事件，再看最后一张画面。成功与否不应该由“看起来差不多到位了”代替。

<a id="part-15"></a>
## 15. 可视化、轨迹与复现

### 15.1 四视角是同一个世界，不是四个环境

[multiview.py](../src/ibero/multiview.py)的 `CableHandoverMultiView` 用四个相机观察同一份 `model/data`，将四张 320×240 图像拼为 640×480 图像。

相机分别跟随总览中心、左 pinch、右 pinch 和主端头。它们是审阅器创建的自由相机，不要把它们与 MJCF 中具名的 `front/left_wrist/right_wrist` 三个相机混为一谈。

`MultiViewDashboard` 再将图像、阶段、抓取、张力曲线与按钮放进 Tk 面板。面板不会创建另外一份物理世界；关闭面板不会停止主环境的物理运行。

### 15.2 为什么回放不继续执行策略

[review.py](../src/ibero/review.py)中的 `Trace` 保存 MuJoCo 的积分状态和逐帧 `info`。回放通过 `mj_setState()` 恢复记录，再 `mj_forward()` 更新画面。

它没有恢复脚本的阶段计数、任务保持计时器等所有 Python 对象。因此这是一份审阅轨迹，不是完整 checkpoint。恢复到某帧后直接继续 `env.step()` 会被拒绝；Enter/reset 才能开始新一轮真实积分。

保存的是控制采样帧，不是每一个 0.0002 s 的子步画面。子步安全峰值与事件由审计结果补充保存。

### 15.3 三类产物不要混用

| 产物 | 用途 | 单独缺少什么 |
|---|---|---|
| PNG | 人工检查安装、位置和接触关系 | 没有完整动态过程 |
| manifest JSON | 记录配置、来源、seed、版本、结果和审计 | 不是逐帧物理轨迹 |
| Trace NPZ | 回放积分状态与逐帧结果 | 不包含完整策略 checkpoint，也不自行打包所有模型资产 |

重放需要相同场景、源码、机器人资产和 MuJoCo 版本。场景哈希覆盖 YAML 内容与任务文件；源码和资产另有哈希。NPZ 读取禁用 pickle，并检查状态维数与有限值。

这些哈希用于发现“这次实验是否还是同一配置/实现”，不是证明它必然物理正确。同一 seed 的重复性也只是受当前实现、依赖和计算环境约束的工程复现，不是跨任意硬件的数学保证。

当前源码哈希读取 `src/ibero` 的 Python 文件；本文这种 Markdown 介绍不会修改物理源码或场景哈希。

<a id="part-16"></a>
## 16. 如何知道实现是可信的

必须将“代码路径正确”“数值模型正确”“真实材料正确”分开看。

### 16.1 第一层：接口和回归

```powershell
python -m pytest -q
python -m ibero.check_scene scenes/cable_handover
python -m ibero.check_scene scenes/cable_stretch
```

测试覆盖旧环境不回退、安装轴、末端切换、reset 可重复、真实夹持、多视图、动作输入、任务成功/失败及轨迹读取等。

### 16.2 第二层：独立力学试验

[materials/bench.py](../src/ibero/materials/bench.py)不依赖机器人完成一个漂亮动作，而是直接构造材料试验台：

- 轴向加载：固定端夹具施加已知伸长，与 `F=kΔL` 的预期比较，并比较不同分段数。
- 质量：检查 Flex 节点质量和是否等于线密度×长度。
- 弯曲梯度：将实际计算的力与能量有限差分比较，检查净力、净力矩。
- 悬垂：在重力下检查支反力与线束自重，提高 EI 后观察下垂是否减小。

这里为了施加已知边界条件而移动固定夹具是合法的材料实验方法；不能因此把同样的直接位置控制偷用作任务中的夹爪抓取。

### 16.3 第三层：整体闭环与故障

```powershell
python -m ibero.verify_core01 --seeds 3 --output artifacts/core01_physics_review.json
python -m ibero.calibrate --output artifacts/core01_stability_20seeds.json
python -m ibero.verify_viewer
```

分别验证材料/传感/多材料闭环与故障、多 seed 保持与交接、实际桌面交互。最后一条会主动创建图形窗口，不适合无图形桌面的普通测试任务。

故障检查故意让夹爪松开、持续外拉或使工装碰撞，确认系统真的会失败，而不是只有成功路径。

### 16.4 编写本文时的证据

| 项目 | 已记录结果 |
|---|---|
| 自动化测试 | 25 项通过 |
| 三组材料 × 三个 seed 的张力闭环 | 9/9 成功 |
| 20 seed × 10 秒保持 | 20/20 通过 |
| 20 seed 的交接 | 18/20，达到本轮 ≥18/20 门槛 |
| 独立轴向刚度检查 | 最大相对误差约 0.063% |
| 悬垂支反力检查 | 最大误差约 3.58% |
| 腕部 5 N 增量静载 | 幅值误差约 1.68% |
| 实际窗口的等待、两轮运行、暂停/单步/回放 | 通过事件驱动检查 |

交接 seed 11 的落点约偏离目标 4.80 cm，超过 4 cm 标准；seed 14 停在接近阶段。两次均记录为超时，没有被宽松判为成功。

这些是窄场景的工程验收结果，不是策略泛化分数。上述力学比较主要证明软件与定义的等效模型一致，真实线束标定仍需测量数据。

<a id="part-17"></a>
## 17. 按问题定位文件与扩展项目

### 17.1 “我想改……”应先去哪里

| 想了解或修改的内容 | 第一入口 | 还要联动检查 |
|---|---|---|
| 命令选哪个场景、参数是什么意思 | [demo.py](../src/ibero/demo.py) | `ibero.__init__.make()` |
| 新环境名称怎么被识别 | [__init__.py](../src/ibero/__init__.py) | 不是添加文件夹就自动注册 |
| 线束长度、质量、刚度、端头 | [张力场景 YAML](../scenes/cable_stretch/scene_config.yaml) | `cable.py`、步长与初始化抓取 |
| Flex 怎样生成、端点如何连接 | [cable.py](../src/ibero/materials/cable.py) | `CableParameters` 与模型编译 |
| 弯曲力和张力从哪里来 | [mechanics.py](../src/ibero/materials/mechanics.py) | `bench.py` 的独立验证 |
| G1 裁剪、夹爪安装、工具坐标 | [g1_upperbody_2f85.py](../src/ibero/robots/g1_upperbody_2f85.py) | 资产、handles、安装与夹持测试 |
| 张力目标、扰动、控制增益 | [张力场景 YAML](../scenes/cable_stretch/scene_config.yaml) | `control/tension.py` |
| 关节动作为什么这样执行 | [resolved_rate.py](../src/ibero/control/resolved_rate.py) | 雅可比、工具坐标、关节限位 |
| 什么时候算抓住、掉落 | [环境文件](../src/ibero/envs/cable_handover.py) | `_grasped*` 与对应任务文件 |
| 成功/失败标准 | [张力任务](../scenes/cable_stretch/task_spec.py)、[交接任务](../scenes/cable_handover/task_spec.py) | constraints 与审计不可混淆 |
| 超拉损伤怎样计时 | [audit.py](../src/ibero/core/audit.py) | 环境每子步 observe 的调用 |
| 一个动作包含多少物理步 | [环境 step](../src/ibero/envs/cable_handover.py) | YAML 的 timestep/control_hz |
| 相机、曲线和界面按钮 | [multiview.py](../src/ibero/multiview.py) | `review.py` 的事件循环 |
| 回放为什么不兼容旧文件 | [review.py](../src/ibero/review.py) | 场景、源码、资产、引擎哈希/版本 |
| 实验是否通过验收 | [verify_core01.py](../src/ibero/verify_core01.py)、[tests](../tests) | artifacts 报告与实施契约 |

### 17.2 当前哪些设置还不是完全数据驱动

不能因为有 YAML，就认为所有细节都已经可以在其中自由替换：

- 双端张力外环的主要增益与阈值来自 YAML；交接脚本中的接近偏移、速度尺度和释放高度等仍在脚本中。
- 抓取历史长度、4 mm 相对位移阈值等目前还在环境代码中。
- 交接裁判的部分最终开度/低速/保持阈值在任务文件中，未全部迁移到 constraints。
- 相机/传感器的具名结构由 P2 构建器提供；YAML 的名称声明不是通用相机设计器，四格审阅视角另由面板代码定义。
- 工装目前支持具体的 receiver 参数，不是任意 CAD 工装布局系统。
- `end_layout` 同时影响材料布局和现有运行分支；完全不同的材料/任务不能保证只改 YAML 即可接入。

这些是理解当前实现的边界，不是建议初学者现在就大规模重构。

### 17.3 增加一个“同类材料变体”的最小流程

1. 复制现有张力菜谱目录，保留三份必要文件和 README。
2. 修改 `id`、材料和来源，必要时调整初始中心距与双臂准备位姿。
3. 执行 `check_scene`，确认声明与保守数值条件满足要求。
4. 通过 `--scene` 指向它进行试跑：

```powershell
python -m ibero.demo --env cable_stretch --scene scenes/my_cable_variant --viewer
```

5. 检查初始接触、工作空间和扰动恢复，保存 manifest，运行相关材料/故障回归。
6. 需要固定的独立环境名称时，再修改 `ibero.make()` 和 demo 选项；不需要名称时可继续用 `--scene`。

新字段需要同时修改 schema 与使用它的代码；直接拼写一个未实现字段不会自动拥有新功能。

### 17.4 修改幅度越大，检查范围越大

只改目标张力时，重点检查力控与安全阈值；改线束刚度/分段数时，还要检查材料步长和稳定性；改端头尺寸或机器人姿态时，还要检查夹持标记、可达性和碰撞。

如果新增扭转、分叉线束、裸线夹持或另一种灵巧手，就已经超出“同类菜谱调参”，需要 P3/P2/环境与测试共同扩展。

<a id="part-18"></a>
## 18. 推荐源码阅读路线与理解检查

### 18.1 不要按文件大小阅读，按数据流阅读

第一遍只看入口与配置：

```text
scenes/cable_stretch/scene_config.yaml
→ constraints.yaml
→ demo.main()
→ ibero.make()
→ CableHandoverEnv.__init__()
```

第二遍看“世界怎样生成”：

```text
SceneLoader.validate()
→ SceneCompiler.compile()
→ build_g1_handover_model()
→ CableParameters.from_scene() / add_cable()
→ HandoverModelHandles / CableHandles
```

第三遍看“世界怎样运动”：

```text
CableHandoverEnv.reset()
→ CableStretchScript.action()
→ BimanualResolvedRateController.apply()
→ CableHandoverEnv.step()
→ CableMechanics.apply() / tension() / state()
→ SafetyAuditor.observe()
→ CableStretchTask.evaluate()
```

最后再看 `review.py`、`multiview.py` 和 `verify_core01.py`，理解怎样观察、记录和证明前面的运行。

### 18.2 读完后，应能回答的八个问题

1. **谁决定线束多长？** 场景 YAML；P3 将它转换为节点/边、质量与力学参数。
2. **谁决定下一步向哪动？** 张力或交接脚本；低层控制器把末端意图转为关节执行目标。
3. **谁让物体真的运动？** MuJoCo 根据执行器、接触、材料力与约束推进动态状态。
4. **谁确认抓住了？** 环境的多条件接触/相对滑移判断，不是脚本发送了闭合命令就算。
5. **谁计算线束张力？** 根据真实 Flex 边长、原长与速度恢复材料边轴向力，不是直接读腕部 F/T。
6. **谁判断成功？** 菜谱的 TaskSpec；安全审计另行记录且本回合损伤不会因卸载而消失。
7. **为什么不是只看 `core/`？** 因为环境编排、P2、P3、控制与 MuJoCo 一起才组成运行核心。
8. **现在是否已经是“真实工业线束数字孪生”？** 不是。它是可配置、可审计、经过数值/工程验证的等效线束操作原型，仍需真实材料与硬件标定。

用一句话收束整个架构：**菜谱描述实验，P2 搭机器人，P3 给材料以物理响应，控制器提出动作，环境组织引擎推进，审计与任务裁判评价结果，审阅工具保存可复查证据。**

<a id="part-19"></a>
## 19. 工业扩展：弹性卡扣、真实孔槽与加工载荷

本节是 S8 开发期的实现导读。卡扣/组合与端铣台架已有阶段证据；机器人三个形状各自完成过完整成功回合，跨种子、变体和最终源码回归仍在闭合。不能把这里的“如何实现”读成所有发布门槛已经通过。

### 19.1 三种对象，三种不同的物理问题

| 对象 | 状态主要是什么 | 力从哪里来 | 什么变化是永久的 |
| --- | --- | --- | --- |
| 线束 | 节点位置、边长、弯曲与端头运动 | 轴向伸长、弯曲恢复、阻尼及接触 | 本阶段不模拟真实断裂；报警会锁存 |
| 弹性卡扣舌片 | 被动分段梁的转角与接触净空 | 梁恢复力和原生接触力 | 正常弹性卸载恢复；不支持塑性/疲劳断裂 |
| 固定钢板毛坯 | 哪些体素还存在、已去除体积和事件版本 | 单独的端铣平均过程模型及非切削面接触 | 已删除的材料不会自行长回来；reset 才回到初始毛坯 |

所以“材料专家”不是把所有东西换成同一个软体引擎。线束需要连续运动和弯曲，卡扣需要可恢复变形与几何互锁，加工需要永久拓扑变化及过程载荷。当前都由 MuJoCo 3.8.1 负责机器人、惯性、接触和积分，项目自己的材料/工艺模块补充各对象缺少的状态与响应。

### 19.2 工业扩展的文件地图

```text
scenes/
  latch_bench/       独立按压/拉拔台架菜谱
  latch_release/     左臂按舌片、右夹爪拔插头
  harness_unplug/    插头携带 Flex 线束，拔出后放入承接托盘
  stock_bench/       几何/实体碰撞检查，不是正常切削入口
  milling_bench/     三平移轴＋真实主轴的过程辨识台架
  plate_milling/     G1 左臂主轴、右臂停安全位、固定毛坯

src/ibero/
  core/             配置、构建分派、外加载荷、审计、轨迹和复现身份
  materials/        参数、弹性梁、线束、毛坯占据及碰撞绑定
  mechanisms/       snap_latch.py：卡扣几何与状态观测
  processes/        刀具扫掠、啮合、平均力、隐式耦合及独立形状检查
  robots/           原 G1 资产裁剪、按压工具、夹爪、主轴与支撑装配
  control/          按压/拉拔策略、加工路径和原执行器阻抗控制
  envs/             机器人任务运行编排
  benches/          独立力学/加工/探针辨识装置
  industrial_review.py   持久窗口交互
  milling_review.py      独立材料显示副本、孔槽俯视、剖面与力图

scripts/            真实长测、GUI 检查、失败诊断和阶段报告汇总
tests/fixtures/     两个真实失败子步的最小数值回归初态
validation/         冻结的门槛与参考资料约定
artifacts/industrial_core01/  实际运行报告、轨迹、截图及保留的失败
```

这是责任划分，不是每个目录都独立构成一个引擎。当前 `plate_milling` 的运行类还复用 `benches/milling.py` 中的共同加工步循环；整理这个共享边界属于后续 S9。并没有预先实现多引擎调度系统。

### 19.3 P3 怎样实现“按下才能拔出”的卡扣

先看 [materials/parameters.py](../src/ibero/materials/parameters.py) 的 `BeamParameters`：长度、宽度、厚度、杨氏模量、密度、弯曲松弛时间均为 SI 单位，并要求填写参数来源。厚度影响截面惯性矩，因此改变厚度不是只改外观。

[materials/elastic_beam.py](../src/ibero/materials/elastic_beam.py) 把舌片建成若干有质量的实体段，相邻段由没有驱动电机的转动关节连接。它们的弹性系数来自梁刚度 `EI`：内部关节约为 `EI/h`，根部半单元为 `2EI/h`，阻尼取转角刚度乘松弛时间。关节由 MuJoCo 被动弹簧和阻尼响应接触载荷，并不是任务脚本指定每段应该弯多少。

[mechanisms/snap_latch.py](../src/ibero/mechanisms/snap_latch.py) 再添加扣齿、插座肩部及导向结构。未按压时扣齿和肩部有几何倒扣，拉拔会产生阻挡接触；按压让舌片弯下，实际净空足够后扣齿才可滑出。`locked/released` 等标签用于解释观测，不是一个用来放行碰撞的开关。过早撤去压头，梁恢复后仍可能再次受阻；已经滑过肩部后撤压则可能合法退出。

当前是一个可验证的、尺寸明确的平面弹性舌片模型，不是某个 RJ45 商品连接器的完整复刻。三维 Flex 候选在独立台架出现数值不稳定，失败数据保留，因此没有把它强行设为默认。默认被动梁通过解析刚度、卸载恢复、步长/分段数和真实接触检查。真实塑料的塑性、疲劳、应力集中和断裂仍需单独数据与模型。

### 19.4 从卡扣台架到真实双臂操作

独立台架允许夹具按规定运动，用于辨识材料与几何；机器人场景不能借台架的直接驱动能力冒充 G1 能力。

[robots/g1_industrial.py](../src/ibero/robots/g1_industrial.py) 固定 G1 基座、保留上身和双臂，左侧安装实际按压工具，右侧安装夹爪。reset 可以用逆解安排初态；运行时由执行器推动机器人，插头仍是动态物体。右夹爪必须通过两侧接触与摩擦抓住它，不能用隐藏绑定把插头焊到手上。

[control/latch.py](../src/ibero/control/latch.py) 组织落稳、抓持、按压、拉出、移开压头与退出。环境在每个物理子步检查材料响应、穿透及安全状态。`task_spec.py` 只判断真实退出和抓持历史是否满足条件，不能在被挡住时直接把插头移出去。

`harness_unplug` 通过 [materials/harness.py](../src/ibero/materials/harness.py) 给插头装上实际 Flex 线束。这里的端部连接属于产品内部的材料装配，不是机器人抓取捷径。拔出后还有降低、开爪、短距离自然落到承接盘及稳定保持。它不等于已经实现重新插入或零跌落的精密装配。

### 19.5 毛坯怎样“真的少了一块”

[materials/stock.py](../src/ibero/materials/stock.py) 的 `VoxelStock` 拥有毛坯初始尺寸、工件坐标系、网格单元、剩余占据、剩余体积/质量、事件列表和版本号。索引始终在工件坐标内计算，相机变化不影响删哪一格。

一次去除先生成候选事件，校验后才提交。相同事件重复应用不会重复扣体积；非法索引、冲突版本和非有限数据会被拒绝。边缘不完整格按真实尺寸计体积。当前采用单元中心是否进入刃区的离散规则，因此毫米网格不能宣称微米尺寸精度。

[materials/stock_collision.py](../src/ibero/materials/stock_collision.py) 为每格预先创建固定实体 box。删料提交时同步材料占据、该格碰撞位和透明度，保留稳定的 geom ID；其它剩余格继续阻挡物体。提交失败会回滚，材料与碰撞版本不一致会立即报错。

这些代理属于固定毛坯，不重复赋予自由刚体质量。剩余质量由材料账本计量；本阶段不支持把切后的钢板松开成为自由运动余料，也不支持从整板切下一个会飞走的块。视觉孔洞、实体碰撞孔洞和自由断片是三种不同能力。

### 19.6 刀具、主轴、刃区和刀柄为何分开

[processes/tools.py](../src/ibero/processes/tools.py) 定义刀尖位姿与端铣刀几何，区分可切圆柱刃区和非切削刀柄。扫掠按网格尺度细分平移与旋转，只返回真正被有效刃区覆盖的候选格；刀柄经过材料不能删除它。

机器人主轴由 [robots/g1_milling.py](../src/ibero/robots/g1_milling.py) 安装到左腕。转子有实际惯量和独立限矩电机，真实转速来自关节速度，不是显示动画或控制命令。新工具不能继承 G1 手臂电机的干摩擦/附加惯量；这曾导致电机根本起不转，已有回归测试。右臂只停安全位，当前任务不是双臂协同铣削。

工作站保持原 G1 关节位置、命令和力矩范围，另设 15 N 加工力包络。刀轴允许小幅真实跟踪误差，但要求角度、角速度和刃区投影误差均在显式范围内；不支持任意五轴。刀具的实际小幅倾斜仍进入几何扫掠，不通过改姿态状态“扶正”。

### 19.7 一次加工子步到底按什么顺序执行

```text
P4 目标 → 路径控制器 → 原机器人执行器命令
                              ↓
                    读取实际刀尖速度与主轴转速
                              ↓
剩余材料占据 → 当前啮合 → 平均切削力/扭矩模型
                              ↓
                独立 MjData 求解力—下一步速度耦合
                              ↓
                检查能力、有效模型范围与数值残差
                              ↓
            提交实际子步扫掠的材料事件＋同步实体碰撞
                              ↓
                施加工具载荷和工件反作用 → mj_step
                              ↓
                    逐步审计 → 只读任务评价
```

负责这一闭环的核心是 [processes/milling.py](../src/ibero/processes/milling.py)。它读取实际位姿/速度与剩余材料，**不读取目标孔槽区域来决定删料**。目标形状只影响上层路径，并用于后续独立检查。

刃区工作时，“刃区—毛坯”的正常切削阻力由过程模型提供，避免同时算硬接触和剪切载荷两份阻力。刀柄、夹具和机器人接触继续有效；静止或未到可切转速时刃区也是实体。豁免的是特定接触对，不是刀具可以穿过所有物体。

超载或耦合求解失败会拒绝下一子步，不继续无条件删料。一个 100 Hz 控制帧可能已完成若干物理子步，然后在下一个子步停止；报告因此分别记录已积分子步数、候选需求和实际施加载荷。

### 19.8 工程过程力与“真实接触力”的区别

[processes/milling_forces.py](../src/ibero/processes/milling_forces.py) 使用转速、进给、齿数和剩余材料啮合估计平均切屑厚度。侧刃剪切系数用 Pa，刃口系数用 N/m；中心下切采用显式端面工程近似。系数写在菜谱里并标注 `AISI1045-provisional`，不是已测钢材数据库。

[processes/prepared_milling_wrench.py](../src/ibero/processes/prepared_milling_wrench.py) 只把同一个子步的几何系数预组合，便于反复计算候选速度对应的载荷。它没有跨物理步缓存目标力，也不降低材料更新频率。

[processes/implicit_wrench.py](../src/ibero/processes/implicit_wrench.py) 解决“力改变速度、速度又改变切削力”的闭环。先在独立 MuJoCo 数据上测响应，求候选速度，再以真实动力学复验。原生干摩擦会使响应分段非线性，必要时直接求真实响应残差；数值折点采用有界方向选择和正则化尝试，原精度与能力门槛仍然保留。

这里的切削力是**作用于真实仿真动力学的工程模型载荷**，不是由钢材微观断裂自动涌现的高保真力。相比之下，夹爪、舌片和夹具之间的阻挡/摩擦来自原生接触求解。两类力可以共同存在，但必须区分来源。

[core/loads.py](../src/ibero/core/loads.py) 也保留这一区别：旧线束继续使用广义力累加；加工的 `PhysicalLoads` 保存刚体世界系作用力、作用点和绕质心力矩，写入 `xfrc_applied`。只写广义力可能让物体运动，却丢失腕部静态 F/T 所需的刚体受力位置。施力端会把作用点力臂加入力矩，并记录固定工件的反作用；固定工件不会因此移动。

### 19.9 成功判定为何要独立于控制器

[scenes/plate_milling/task_spec.py](../scenes/plate_milling/task_spec.py) 定义直槽、浅腔和通孔目标。`MillingBenchScript` 用它生成进刀、加工、退刀、停主轴路径。机器人内层 `MillingArmServo` 则把位姿误差和实际速度转换为原执行器命令，补偿重力并检查能力；运行时不直接写关节位置。

路径完成之后，[processes/shape_check.py](../src/ibero/processes/shape_check.py) 独立检查体积、边界、少切/过切、内部覆盖和贯通列。只达到相同体积、却切错位置，不能成功。环境还检查实际退刀、实际停转及是否发生安全失败，最后交给只读 P4 任务判定。

[benches/stock_probe.py](../src/ibero/benches/stock_probe.py) 再从**加工后的实际占据**建立独立动态探针检查：通孔应可通过，槽底/浅腔底应阻挡，邻近剩余实体仍应阻挡。它不是机器人在线持测头，也不是先按目标画一个理想孔来验证自己。

### 19.10 变形回看与材料回看不完全相同

弹性梁的形变主要在关节状态中；毛坯的孔洞还在材料事件和碰撞模式中。因此加工用 [core/stock_trace.py](../src/ibero/core/stock_trace.py) 同时记录物理积分状态、材料事件前缀、材料 hash 与刃区模式。

加载会核对场景、所选目标、源码、资产和引擎身份，并限制归档体积/条目/数据类型；不使用 pickle。只回看状态，不恢复策略、求解器历史等所有续跑上下文，所以回放后必须 reset，不能把它当成训练 checkpoint。

[milling_review.py](../src/ibero/milling_review.py) 的显示端拥有独立 `MjModel + MjData + VoxelStock + StockCollisionBinding`。回看旧孔、隐藏工具和开关视觉切屑只操作显示副本，不改变 live 世界。四格依次看全景、刀具局部、毛坯俯视及占据剖面/全回合力峰值。默认 fast 关闭阴影和地面反射，可用 `--render-quality quality` 恢复；两者没有不同的物理精度。

### 19.11 从哪里运行，怎样避免误读验收

```powershell
python -m ibero.demo --env latch_release --viewer
python -m ibero.demo --env harness_unplug --viewer
python -m ibero.demo --env plate_milling --shape through_hole --viewer
python -m ibero.demo --env plate_milling --shape pocket --viewer --no-visual-chips

python scripts/verify_g1_milling_preflight.py
python scripts/verify_milling_process.py --scene scenes/plate_milling --shape all
python scripts/verify_robot_milling.py --workers 3
```

最后三条分别是静态包络、单种子全链和三形状各十种子的检查，不互相代替。GUI 检查需要显式运行；无头数值通过不能声称窗口已验收。完整工业统一验收入口的收敛仍在 S9，现有 `ibero.verify_industrial` 不能被误解为所有扩展已一键验完。

加工环境目前由专用 demo 入口构造，不冒充与旧线束完全相同的 Gym 动作接口。标准化训练、LeRobot、真实 SDK 和策略采集没有在这一轮提前铺开。

查看结果应同时看 `report.json` 的版本、输入、分母和失败原因，`trace.npz` 的实际过程，以及 `sections.png`/四视图和动态探针。实时因子统一按仿真时间除以墙钟时间报告；当前细步长工业仿真明显慢于实时，不用加速回放的观感代替这个数字。

### 19.12 当前扩展到底证明了什么

它证明项目已能把**材料状态、可解释的力学/过程响应、实际机器人执行器、碰撞、独立任务裁判与可复查证据**连成具体闭环，而不只播放“按一下卡扣”或“钢板上出现一个洞”的动画。

尚未证明的包括真实商品连接器精度、真实钢材加工力标定、任意三维弹性体、真实断裂/自由余料、热切屑、刀具磨损、颤振、五轴及实机安全。这些不是补一个 YAML 字段就自动得到的能力，也不因为三个场景可运行就进入 0.2。工程门槛与真实标定状态始终分开登记。
