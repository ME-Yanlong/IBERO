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

        for _ in range(15):
            r = residual(u)
            if np.linalg.norm(r * scale) < 1e-10:
                break
            jac = np.eye(4)
            for j, epsilon in enumerate((1e-7, 1e-7, 1e-7, 1e-3)):
                du = np.zeros(4)
                du[j] = epsilon
                jac[:, j] = (residual(u + du) - residual(u - du)) / (2 * epsilon)
            delta = np.linalg.solve(jac, -r)
            for factor in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                candidate = u + factor * delta
                if candidate[3] > 1 and np.linalg.norm(
                    residual(candidate) * scale
                ) < np.linalg.norm(r * scale):
                    u = candidate
                    break
            else:
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
