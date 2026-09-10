# 自由卡扣拔出：S4 开发中

入口：`python -m ibero.demo --env latch_release --viewer`。Enter 新回合，P 暂停，N 单步，B 后退，R 只读回放。窗口结束后保留；三路物理相机加一个指标面板，拖动视图可旋转，滚轮缩放。

左臂真实按压工具、右臂 2F-85，保留原 G1 关节力矩上限。插头是自由物体，舌片采用已通过 G3 台架的被动弹性梁；按压使扣齿实际几何让开肩部，不切换约束，不写插头位姿。先释放扣齿，再撤开按压工具，最后完全退出 75 mm 并稳定保持 0.25 s。

`scene_config.yaml` 管材料、机器人、初始化和控制参数；`constraints.yaml` 管安全与任务门槛；`task_spec.py` 只读实际物理状态判定。seed 真的改变插座位置和摩擦。示例材料尚无实测标定。

单次 seed 0 已成功，但完整多种子验收及带线束的 `harness_unplug` 尚未通过，不代表 G4 完成。物理步长 2.5 μs，逐子步安全审计；当前单次 10.29 s 仿真约耗 423 s，约 0.024 倍实时，未达 0.5 性能目标。回放可以流畅播放，但不是实时求解能力。

检查入口：`python scripts/verify_latch_robot.py --workers 4`，保存全部成功/失败 JSON、NPZ 和源码/资产/场景标识；验收运行期间不得修改其源码。`python scripts/check_industrial_viewer.py` 只检查短程交互，不替代两个完整成功回合的 GUI 验收。
