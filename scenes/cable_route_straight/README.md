# 直线多卡扣布线

三只相同的弹性侧向线夹沿板面排成一行。任务成功要求同一根线束按材料弧长顺序经过全部线夹，双手离开后保持稳定。线束和夹臂物性均为示例参数，尚未实测标定。

```powershell
python -m ibero.check_scene scenes/cable_route_straight
python -m ibero.demo --env cable_route_straight --seed 0 --viewer
```
