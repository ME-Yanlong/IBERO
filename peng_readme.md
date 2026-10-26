conda activate ibero-core
E:
cd E:\交大\交大科研\【C】 项目\IBERO

python -m ibero.demo --env cable_stretch --viewer --playback-rate 0.5


# 新增：按压卡扣并拔出
python -m ibero.demo --env latch_release --viewer --seed 0

# 新增：带线束插头拔出并放置
python -m ibero.demo --env harness_unplug --viewer --seed 0

# 新增：加工通孔
python -m ibero.demo --env plate_milling --shape through_hole --viewer --seed 0

# 加工直槽
python -m ibero.demo --env plate_milling --shape slot --viewer --seed 0

# 加工浅腔
python -m ibero.demo --env plate_milling --shape pocket --viewer --seed 0


# Core-0.2 布线任务：单夹和三夹直线已完成 seed 0 演示
python -m ibero.demo --env cable_clip_insert --viewer --seed 0
python -m ibero.demo --env cable_route_straight --viewer --seed 0

# 菜谱贡献样例：线夹横移 6 mm，仍复用同一物理 Core
python -m ibero.demo --env cable_clip_insert --scene scenes/cable_clip_insert_offset --seed 0

# L 形、开口矩形仍是可运行的候选任务；当前参考控制器会报告弯曲半径失效
python -m ibero.demo --env cable_route_l --seed 0
python -m ibero.demo --env cable_route_rectangle --seed 0

# 列出任务菜谱；查看四场景快检报告；离线视频网页在下方路径
python -m ibero.task_catalog
python -m ibero.verify_routing --suite quick
# artifacts/core02_showcase/site/index.html
