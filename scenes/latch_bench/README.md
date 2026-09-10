# 弹性卡扣材料台架（非机器人任务）

这是 G1 接入前的材料辨识夹具。插头有明确的轴向滑台边界，按压头有独立竖直滑台；两个限力位置致动器驱动夹具。真实接触决定是否可以退出，没有“按下即关闭锁”的开关。

入口：`python -m ibero.verify_industrial --suite latch`。输出包含三组参数、六种动作反事实、端载梁解析比较、数值收敛与失败的 Flex 候选。它不验证机器人抓取，也不是完整工业版本验收。

## 从菜谱追到物理

`scene_config.yaml` → `SceneLoader.validate` → `core/industrial_config.py` → `SceneCompiler.compile` → `benches/latch.py:build_latch_fixture` → `mechanisms/snap_latch.py:add_latch` → `materials/elastic_beam.py:add_elastic_beam` → MuJoCo 模型。

批量验收使用相同 builder，通过 Python 参数产生受控变体；不会假装 CLI 自动读取用户任意改动后的菜谱。需要单独使用菜谱时调用 `SceneLoader` 和 `SceneCompiler`。`constraints.yaml` 的上限应传给运行时观察器；`CompiledFixture` 只负责编译模型，不运行任务。

局部坐标：舌片根部为原点，舌片朝 +X 延伸，拔出沿 −X，按压沿 −Z，宽度沿 Y。默认舌片 60×12×2 mm；扣齿高 6 mm、轴向长 4 mm；肩部初始重叠 3 mm、轴向间隙 1 mm。压头位于根侧 X=40 mm，半宽 0.8 mm；不穿过肩部接触舌片。自由机器人版本不能沿用插头滑台。

## 模型与有效范围

矩形梁 `EI=E*b*t³/12`；段长 h=L/N，转角弹簧 k=EI/h，根部半单元 k=2EI/h；阻尼 c=k*τ。示例 E=2 GPa、ρ=1200 kg/m³、τ=3 ms。此降阶模型主要描述面内弯曲，不包含扭转、局部压溃、塑性、疲劳、断裂，也没有真实产品标定。

求解器接触参数用于控制离散接触误差，不是材料杨氏模量。2.5 μs 步长用于处理薄梁—扣合的高频响应，性能可能明显低于实时；验收不隐藏墙钟耗时。初始尝试和失败见开发文档 §18.3。

`LatchObserver` 只读扣齿/肩部的实际包围几何、梁端位移、接触力和数值警告；状态标签不反过来施力。净空标签有 20 μm 防抖，真实接触不使用此标签。超过力/挠度范围锁存 `invalid_reason`，只能 reset 清除；无断裂模型时不称为“已模拟破坏”。

## 状态与生命周期

| 状态 | 唯一拥有者 | 恢复方式 |
| --- | --- | --- |
| 几何、质量、关节被动系数、求解器设置 | `MjModel`，由菜谱和 builder 生成 | 参数改变重新编译，不能沿用旧整数句柄 |
| 位姿、速度、致动器输入、接触结果、时钟、外载 | `MjData` | `mj_resetData` 后 `mj_forward` |
| 超限锁存、事件、接触峰值、标签防抖 | `LatchObserver` | `reset()` 后读取初始状态 |
| 任务进度/成败 | `task_spec.py` | 任务 reset 后读取物理状态，不写引擎 |
| 每子步外加载荷缓冲 | `AppliedLoads` | 每步 begin 清空，全部模块累加后 commit 一次 |
| 图形上下文 | 调用者持有的 renderer/viewer | 调用者显式 close，台架无隐式窗口 |

当前卡扣无额外可变拓扑状态。后续 stock 的材料占据/事件需要独立拥有者及同步事务，不能假设 `mj_resetData` 会恢复孔洞。

参考数据示例在 `validation/references/beam_analytical.*`，明确标为解析值。CSV 导入只接收 SI 单位；拟合训练索引与留出索引不重叠。实测来源标签不等于平台自动认证该数据的真实性。
