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
