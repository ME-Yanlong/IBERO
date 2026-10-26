# 非默认布线贡献样例：线夹横移 6 mm

这是 `cable_clip_insert` 的菜谱变体。它复用相同的 G1、自由 Flex、侧入式弹性线夹和独立任务判据；只改 `scene_config.yaml` 的场景 ID、线夹/目标位置，以及薄 `task_spec.py` 的说明。无需修改物理 Core。

```powershell
python -m ibero.check_scene scenes/cable_clip_insert_offset
python -m ibero.demo --env cable_clip_insert --scene scenes/cable_clip_insert_offset --seed 0
```

默认材料数值仍是 `provisional`。此菜谱只展示可贡献性和相同种子可重放，不能代替参数范围的泛化验收。
