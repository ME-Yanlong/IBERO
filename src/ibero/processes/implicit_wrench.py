"""平均加工载荷的隐式子步耦合；所有预测在独立 MjData，绝不写回预测位姿。"""

import mujoco
import numpy as np


class ImplicitWrenchCoupling:
    """测量当前 MuJoCo 子步对六维外载的速度响应，再解四维非线性负载方程。

    未把力按经验缩小：求的是 u_next = u_free + C W(u_next)，其中 u 包括
    刀尖平移速度和实际主轴角速度。求得的 W 通过 xfrc_applied 在实际刚体施加。
    接触/强非线性使该局部响应失效时拒绝本步，不隐藏失败继续删除材料。
    """

    def __init__(self, model, *, tool_body, tip_site, spindle_dof):
        self.model = model
        self.body, self.site, self.spindle_dof = tool_body, tip_site, spindle_dof
        self.scratch = mujoco.MjData(model)
        self.last_diagnostics = {}

    def _trial(self, data, base_generalized, base_body, point, wrench):
        m, d = self.model, self.scratch
        mujoco.mj_copyData(d, m, data)
        d.qfrc_applied[:] = base_generalized
        d.xfrc_applied[:] = base_body
        d.xfrc_applied[self.body, :3] += wrench[:3]
        d.xfrc_applied[self.body, 3:] += wrench[3:] + np.cross(
            np.asarray(point) - d.xipos[self.body], wrench[:3]
        )
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.site, velocity, 0)
        result = np.r_[velocity[3:], d.qvel[self.spindle_dof]]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite implicit milling prediction")
        return result

    def solve(self, data, base_generalized, base_body, point, wrench_at_velocity):
        self.last_diagnostics = {}
        free = self._trial(data, base_generalized, base_body, point, np.zeros(6))
        compliance = np.empty((4, 6))
        for i in range(6):
            basis = np.zeros(6)
            basis[i] = 1
            compliance[:, i] = (
                self._trial(data, base_generalized, base_body, point, basis) - free
            )
        u = free.copy()
        scale = np.array([1, 1, 1, 0.001])  # 速度残差以 m/s、主轴残差以千 rad/s 归一。

        def residual(value):
            if value[3] <= 1 or not np.isfinite(value).all():
                raise ValueError("Spindle stalled in implicit cutting prediction")
            wrench = wrench_at_velocity(value)
            return value - free - compliance @ wrench

        # 连续切削优先用实际上一时刻速度作初猜；自由预测在强负载时可能远离根。
        current_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.site, current_velocity, 0
        )
        current = np.r_[current_velocity[3:], data.qvel[self.spindle_dof]]
        if current[3] > 1 and np.linalg.norm(
            residual(current) * scale
        ) < np.linalg.norm(residual(u) * scale):
            u = current
        for iteration in range(40):
            r = residual(u)
            self.last_diagnostics = {
                "iterations": iteration,
                "scaled_residual": float(np.linalg.norm(r * scale)),
                "velocity_and_spindle": u.tolist(),
                "residual": r.tolist(),
                "free_response": free.tolist(),
            }
            if np.linalg.norm(r * scale) < 1e-10:
                break
            jac = np.eye(4)
            forward, backward = np.eye(4), np.eye(4)
            for j in range(4):
                # 导数扰动随当前速度缩放；固定 0.1 μm/s 扰动会跨越低速刃口过渡区。
                epsilon = (
                    max(abs(u[j]) * 1e-6, 1e-11)
                    if j < 3
                    else max(abs(u[j]) * 1e-7, 1e-5)
                )
                du = np.zeros(4)
                du[j] = epsilon
                rp, rm = residual(u + du), residual(u - du)
                jac[:, j] = (rp - rm) / (2 * epsilon)
                forward[:, j], backward[:, j] = (rp - r) / epsilon, (r - rm) / epsilon
            # 端面只抗向下运动，零轴向速度处是物理分段条件；尝试单侧广义导数，
            # 不用一条跨越启停面的中心差分把求解卡在边界。方程/变量同时按单位缩放。
            accepted = False
            for derivative in (jac, forward, backward):
                normalized = derivative * scale[:, None] / scale[None, :]
                try:
                    delta = np.linalg.solve(normalized, -r * scale) / scale
                except np.linalg.LinAlgError:
                    continue
                for factor in (2.0 ** (-i) for i in range(16)):
                    candidate = u + factor * delta
                    if candidate[3] > 1 and np.linalg.norm(
                        residual(candidate) * scale
                    ) < np.linalg.norm(r * scale):
                        u, accepted = candidate, True
                        break
                if accepted:
                    break
            if not accepted:
                raise ValueError("Implicit cutting solve did not descend")
        if np.linalg.norm(residual(u) * scale) >= 1e-8:
            raise ValueError("Implicit cutting solve did not converge")
        wrench = wrench_at_velocity(u)
        # 第八次独立预测复验真实响应；没有用线性近似自身充当通过证据。
        actual = self._trial(data, base_generalized, base_body, point, wrench)
        error = float(np.linalg.norm((actual - u) * scale))
        if error > 1e-6:
            raise ValueError(
                "Cutting velocity response is nonlinear or contact-constrained"
            )
        return wrench, actual, error
