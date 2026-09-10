"""三轴台架 baseline：只下发伺服目标，材料删除不读取这些路径。"""

import numpy as np
from ibero.processes.shape_check import MachiningTarget


def bench_target(shape):
    return MachiningTarget(
        shape,
        (0, 0),
        (0, 0)
        if shape == "through_hole"
        else (0.004, 0.002 if shape == "pocket" else 0),
        0.004,
        0.004 if shape == "through_hole" else 0.002,
    )


class MillingBenchScript:
    """进刀→实际轨迹加工→退刀→停主轴；轨迹结束不是形状合格结论。"""

    def __init__(self, env, target):
        if env.scene is None:
            raise ValueError("Controller requires a declared milling recipe")
        self.config = env.scene.config["control"]
        self.dt = 1 / env.scene.config["physics"]["control_hz"]
        self.target_shape = target
        a, b = target.half_straight_xy_m
        x, y = target.center_xy_m
        top = env.stock.params.size_m[2] / 2
        bottom = top - target.depth_m
        # 通孔略穿出板底；实体去除仍只限于实际毛坯，不制造额外材料。
        cut_z = bottom - (
            env.stock.cell_size_m / 2 if target.shape == "through_hole" else 0
        )
        safe_z = top + 1.5 * env.stock.cell_size_m
        self.waypoints = [
            (np.array([x - a, y - b, safe_z]), "approach"),
            (np.array([x - a, y - b, cut_z]), "plunge"),
        ]
        if target.shape == "pocket" and min(a, b) <= env.tool.radius_m:
            # 小浅腔的整个内部都在边框刀具扫掠半径内，闭合四边已覆盖，免去重复满栅格空切。
            for px, py in ((a, -b), (a, b), (-a, b), (-a, -b)):
                self.waypoints.append((np.array([x + px, y + py, cut_z]), "cut"))
        elif target.shape != "through_hole":
            # 密集蛇形覆盖圆角矩形；没有以目标体素区域直接删除材料。
            rows = np.linspace(
                -b, b, max(1, int(np.ceil(2 * b / (env.stock.cell_size_m / 2))) + 1)
            )
            for i, offset in enumerate(rows):
                start_x, end_x = (-a, a) if i % 2 == 0 else (a, -a)
                if i:
                    self.waypoints.append(
                        (np.array([x + start_x, y + offset, cut_z]), "cut")
                    )
                self.waypoints.append((np.array([x + end_x, y + offset, cut_z]), "cut"))
        end = self.waypoints[-1][0].copy()
        end[2] = safe_z
        self.waypoints.append((end, "retract"))
        self.nominal = env.data.site("mill_tip").xpos.copy()
        self.index = 0
        self.phase = "spin_up"
        self.finished = False
        self.stop_started = None

    def command(self, env):
        c = self.config
        actual = env.data.site("mill_tip").xpos.copy()
        if env.data.time < c["settle_seconds"]:
            return self.nominal.copy(), c["rpm"]
        if self.index >= len(self.waypoints):
            self.phase = "stop_spindle"
            if self.stop_started is None:
                self.stop_started = env.data.time
            rpm = abs(env.process.kinematics(env.data)[3])
            self.finished = rpm < 1 and env.data.time - self.stop_started > 0.2
            return self.nominal.copy(), 0.0
        waypoint, self.phase = self.waypoints[self.index]
        distance = np.linalg.norm(waypoint - self.nominal)
        speed = c["plunge_m_s"] if self.phase == "plunge" else c["feed_m_s"]
        self.nominal += (waypoint - self.nominal) * min(
            1, speed * self.dt / max(distance, 1e-12)
        )
        # 保留真实受载跟踪滞后，并等待实际刀尖到位；不以滞后过程力构造正反馈补偿。
        if (
            np.linalg.norm(self.nominal - waypoint) < 1e-10
            and np.linalg.norm(actual - waypoint) < c["tracking_tolerance_m"]
        ):
            self.index += 1
        return self.nominal.copy(), c["rpm"]
