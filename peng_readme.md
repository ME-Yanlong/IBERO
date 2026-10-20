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