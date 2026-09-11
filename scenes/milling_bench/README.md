# 受限端铣过程台架（S7，非机器人任务）

用途是先验证“实际刀具运动 → 材料啮合 → 力/主轴负载 → 能力检查 → 去除事务 → 下一物理步”。三个平移伺服和真实旋转关节都由 MuJoCo 积分；没有在 step 中覆写刀具位置。这里明确采用零重力辨识，不代表 G1 已能加工。

`scene_config.yaml` 由 P4 配置毛坯尺寸/密度、网格、刀具有效刃长和刀柄、材料过程系数、转速/进给及能力范围。所有钢材系数均为 provisional，既不是钢材实测数据库，也不包括温升、颤振、微观断裂和飞屑碰撞。`constraints.yaml` 定义跟踪/刀柄接触限制和验收容差。`task_spec.py` 不会把台架运动冒称机器人任务成功。

实现索引：

- `core/milling_config.py` 严格校验；`benches/milling.py:MillingFixture.from_scene` 建立固定毛坯、伺服和主轴。
- `processes/milling.py` 从剩余体素估计啮合，侧刃采用圆周平均力；中心面刃采用单独声明的工程近似。
- `processes/implicit_wrench.py` 在独立预测状态上解载荷—速度闭环，再将求得的实际刚体力交给 MuJoCo；不会用预测位姿替换实际状态。
- `core/loads.py:PhysicalLoads` 保留力的作用位置，腕部 F/T 能感受到传递载荷。刀具与固定毛坯受到相反的过程力；夹具承担反作用。
- `materials/stock_collision.py` 原子更新剩余体素、双方碰撞位与外观。只对已授权切削的刃区替代硬接触，刀柄/未切实体/其他接触保持有效；停转没有去除授权。
- `control/milling_bench.py` 只下发实际伺服目标。`processes/shape_check.py` 独立比较实际孔槽与解析目标，拒绝少切、过切、偏切；目标区域不会直接驱动材料删除。
- `core/stock_trace.py` 保存物理状态及去除事件前缀；回看不能直接继续积分，必须 reset。

开发中的三形状检查入口（失败会返回非零且保存原始记录）：

```powershell
python scripts/verify_milling_process.py --shape all --workers 3
```

当前网格 1 mm、物理上限 60000 格。该上限不是建议容量：初编译和隐式耦合仍有显著开销。一步内位移、主轴/力/功率或数值求解超限会确定地中止，不通过裁掉载荷后继续删除材料来掩盖问题。完整 G7 与后续 G1 加工状态以开发文档和冻结报告为准。

`constraints.yaml` 的 `max_seconds` 由共同物理步进强制执行；最后一批子步可被截短，不足一个完整物理步则停止，不为填满时长而改变积分步长。超时后必须 reset，不依赖调用者是否记得检查外层循环。

## 参数如何变成力与孔

毛坯是固定夹具中的材料，不是可自由运动的钢块。`density_kg_m3` 决定去除质量账目；固定夹具承担反力，不能通过改密度自动得到更硬的钢材。切削阻力来自单独声明的过程系数；它们必须有牌号、单位与 provisional 状态，不能由字符串 `steel` 自动推断。当前不模拟断下余料的自由运动。

侧刃在刀尖坐标中取刀轴为 +Z，正主轴沿 +Z 转动。径向单位向量 `n=(cosφ,sinφ,0)`，每齿进给厚度 `h=max(v·n,0)/(rpm/60×齿数)`；φ 的零点是本实现的 +X，不直接混用其他资料的角度零点。沿每个角度查询剩余体素列，得到实际啮合刃长 `a(φ)`；每刃切向分量为 `a×(Ktc×h+Kte)`，径向、轴向分量分别使用自己的系数。对一周平均、乘齿数，并计算作用点关于刀尖的力矩。切屑厚度使用**实际运动速度**，不是 baseline 目标速度。

上述形式参照 [UBC/MAL 机加工课程](https://www.malinc.com/wp-content/uploads/2015/06/MAL_Machining_Course.pdf) 的机械式切削模型；进给、转速、扭矩、功率的量纲可对照 [Sandvik 铣削公式](https://www.sandvik.coromant.com/en-gb/knowledge/machining-formulas-definitions/milling-formulas-definitions)。引用模型形式不代表这些示例系数已经获得材料实测支持。

端面只在已声明 `center_cutting: true` 且实际向下进给时工作。以等面积圆盘查询端面剩余材料比例，并使用独立 `face_*` 系数沿径向积分轴向力和扭矩。这是平底中心刃的过程级近似，未表示真实横刃几何、局部崩刃或偏心接触瞬态。

数值与材料参数不要混淆：

- `cell_size_m` 是几何分辨率。用不超过一格的前视估计连续平均啮合，可能在离散边界前出现平均载荷；材料仍只按实际当前/下一物理步运动扫掠去除，不预先挖掉前视区域。
- `edge_transition_chip_m` 当前 10 nm，仅平滑常数刃口项在零进给处的跳变，权重为 `h/(h+δ)`。它不是实测最小切屑厚度，不缩小剪切系数，也不截断过载力。解析参考保留 δ=0，并比较 10/5 nm。
- 默认物理步长现为 100 μs；200 μs 曾出现混合刃区启停求解失败。采用更细步长仍需成组验证，不能靠动画播放速度说明稳定性。
- `machine` 是台架自身的轴伺服刚度/阻尼/力限值。当前 50000 N/m、350 N·s/m、每轴 50 N；它不是 G1 的关节能力。普通 Python 诊断构造器仍保留显式的早期台架默认值，正式样例以加载后的完整菜谱和 manifest 为准。

## 查看哪些审计值

每帧既有物理时刻 `time_s`，也有最后一次载荷求值时刻 `load_evaluation_time_s`。`force_world_n`、`torque_world_nm` 为最后一次过程力及关于刀尖的力矩；`peak_*_step_*` 保存控制帧内逐物理子步峰值，不能只看最后一帧判断是否超限。

`actual_rpm` 来自引擎关节速度；`spindle_power_w` 是模型切削功率，`motor_mechanical_power_w` 是执行器实际机械功率，不是电功率。加减速时二者不必相等，转子惯性会储存/释放能量。`ft_force_n`、`ft_torque_nm` 为安装座传感器坐标中的读数；不能直接与世界系力逐轴比较，需要旋转和力矩臂换算。夹具反作用同时按刀尖和工件原点列出，避免换了参考点却沿用原力矩。

轨迹 v2 同时保存材料事件和刃区接触模式；从原始、中间、最终帧回看都不允许继续积分。更换源码、工具、过程系数、步长或场景后，旧文件按身份校验拒绝；不是自动兼容或可恢复训练 checkpoint。

独立数值覆盖会写入新的证据目录，不改原菜谱，例如：

```powershell
python scripts/verify_milling_process.py --shape slot --timestep-s 0.00005
python scripts/verify_milling_process.py --shape slot --edge-transition-chip-m 5e-9
python scripts/verify_milling_loads.py
```

最后一个入口覆盖合成平均力、非零力臂、旋转 F/T、纯力矩、反作用平衡及边界测试；即使通过，也不等于三个实际加工形状或机器人任务已通过。

## 开始长测前检查网格是否能表达目标

```powershell
python scripts/forecast_milling_geometry.py --scene scenes/milling_bench
```

这个快速入口只对名义目标的体素中心分类及体积权重求和，不建动力学、不提交材料事件。默认比较 1、0.75、2/3、0.5 mm，不能把它当成实际切削、碰撞或力的收敛检查。

例如 4 mm 厚板上的 2 mm 深槽与 0.75 mm 格距不对齐；名义槽/浅腔体积误差约为 8.68% / 11.10%，已经超过 5% 标准。增加计算时间不会自动修复这个离散表示问题。更细也不保证每个固定几何的体积误差单调下降；需要同时检查格距、几何相位、碰撞容量和实际动力学。原 0.75 mm 全链运行会保留，2/3 mm 对齐组另存，不用调整裁判阈值消除失败。
