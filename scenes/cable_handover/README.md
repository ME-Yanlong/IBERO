# Cable handover (Core-0.1)

## 末端执行器选择

`scene_config.yaml` 的 `robot.end_effector.type` 是 P2 的显式构型开关：

- `robotiq_2f85`（默认、已验收）：在挂载两个 2F-85 前移除 vendored G1 模型
  的两个固定 `rubber_hand` 视觉几何。因此场景中不会再出现“G1 原手与夹爪
  叠加”的错误构型；当前接触式 handover baseline 只对此选项有效。
- `g1_native_hand`（仅构型审阅）：保留当前 G1 资产自带的静态手部外观，不挂载
  Robotiq。该资产没有手指关节、致动器、指垫碰撞或灵巧手控制接口，故
  `CableHandover-v0` 会明确拒绝运行，而不会将它误称为可抓取的灵巧手。

后续若引入带自由度、碰撞资产和控制映射的真实 G1 Dex3/Dex5 手，应作为新的
`end_effector.type`、动作接口与独立接触回归实现，不能复用这项视觉选项来声称
已完成灵巧手支持。

The left 2F-85 closes on the green zone of a rigid handover terminal. The
right 2F-85 approaches the yellow zone, closes, the left opens, and the right
places the terminal into the target region. The orange cable is physically
connected to the terminal and its tail; it is neither a wrist anchor nor a
mocap-followed visual.

All cable dimensions, mass, bend radius, terminal geometry, stiffness,
damping, friction and parameter provenance are in `scene_config.yaml`. Replace
the provisional values with a target harness datasheet/measurements before
making a physical-fidelity claim.

This recipe has material fields and explicitly selects `flex`. MuJoCo's 1-D
Flex renders/contact-tests a chain of capsule elements and connects its two
ends physically to dynamic terminals. The old `capsule_chain` alias is rejected:
it was the same Flex implementation, not an independent fallback backend.

The values remain `provisional`. Axial stiffness/damping act on actual Flex
edges, and `bending_stiffness_nm2` supplies an energy-consistent centerline
bending force. `minimum_bend_radius_m` triggers an alarm, not a hard bend stop.
Torsion, cable self-contact and physical fracture are not modeled. Do not
interpret these independent numerical checks as production-harness calibration.

## 本轮物理补强

安装变换将 2F-85 工具 +Z 对准 G1 手腕 +X，双臂使用屈肘准备位姿；没有
修改 vendor 源模型。转接偏移仍是仿真构型，需要真实转接板尺寸后再标定。

交接端头长 0.36 m，是提供两个分离夹持区的校准长条，不代表真实连接器。
夹持要求两侧指垫接触、开度、位置窗口及连续三帧低相对滑移同时满足。
右臂到达托台上方约 0.10 m、减速后张开并退出，端头靠重力落入真实托台。
只有左右手均释放、右夹爪张开、端头距目标不超过 0.04 m、实际支撑接触且
速度低于 0.04 m/s 连续 0.5 秒，才成功。这是接触式交接与重力辅助落料，
不是已经完成轻放/插接控制；后者需要后续接触法向力控制。

`scene_config.yaml` 的 `workcell.receiver` 定义托台偏移与半尺寸。
大幅修改臂姿态、端头或工装时，必须重新检查可达性、净空和接触抓取，
不能认为只通过 YAML 格式校验就保证所有新几何任务能成功。

Validate the recipe with:

```powershell
python -m ibero.check_scene scenes/cable_handover
```

## 人工审阅 demo

```powershell
python -m ibero.demo --env cable_handover --viewer
```

MuJoCo viewer 和弹出的四视角面板都会保持打开。先检查 reset 状态，再在任一
窗口按 Enter（或点击“运行一轮”）执行一次 scripted rollout；结束后最终状态会
保持，下一次 Enter 会用相同 `--seed` 重新运行。四格依次为总览、左末端、右末端、
端头/目标区，并显示阶段、夹持、即时/峰值拉力和安全标志。`--playback-rate 0.5`
可以降低播放速度；四格面板不可用时降级为仅 MuJoCo viewer，但 viewer
本身仍需要图形桌面。P 暂停、N 单步、R 回放、B 上一帧。

不打开窗口也可保存同一组四视角作为 review 证据：

```powershell
python -m ibero.demo --env cable_handover --save-multiview artifacts/cable_handover_4view.png
```
