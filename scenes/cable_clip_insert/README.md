# 双臂单线夹布线任务

这是一根自由 Flex 线束、立式板和弹性侧向线夹。线束没有机器人焊接约束。双臂应抓住两端附近的裸线，向板面压入中间线段，然后松手退离。

`scene_config.yaml` 给出物性、夹口和工位尺寸；`constraints.yaml` 给出载荷与穿透限制；`task_spec.py` 声明由实际状态判定的任务目标。所有材料数据暂为示例参数，尚未实测标定。

建议先运行 `python -m ibero.check_scene scenes/cable_clip_insert`，再运行 `python -m ibero.demo --env cable_clip_insert --seed 0`。此命令在 Core-0.2 布线环境接通后才能执行。
