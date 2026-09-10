"""卡扣的独立按压/拉出夹具，驱动通过位置致动器与力限幅实现。"""

from dataclasses import asdict
import mujoco
import numpy as np

from ibero.core.scene_compiler import fixture_spec
from ibero.materials.parameters import BeamParameters, LatchParameters, finite_number
from ibero.mechanisms.snap_latch import add_latch, LatchObserver


def build_latch_fixture(
    beam=None, latch=None, *, segments=8, timestep=0.0000025, press_x=None, press_y=0.0
):
    beam, latch = beam or BeamParameters(), latch or LatchParameters()
    finite_number(timestep, "timestep")
    if not np.isfinite(
        [beam.length_m * 2 / 3 if press_x is None else press_x, press_y]
    ).all():
        raise ValueError("Press coordinates must be finite")
    cfg = {
        "id": "elastic_latch_fixture",
        "physics": {"timestep_s": timestep, "gravity_m_s2": [0, 0, 0]},
    }
    spec = fixture_spec(cfg)
    spec.compiler.degree = False
    spec.option.iterations = 100
    names = add_latch(spec, beam, latch, segments)
    # 压头位于肩部根侧，不能让压头穿过肩部后再接触舌片。
    pusher = spec.worldbody.add_body(
        name="press_tool",
        pos=[press_x if press_x is not None else beam.length_m * 2 / 3, press_y, 0.018],
    )
    pusher.add_joint(
        name="press_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        damping=2.0,
    )
    pusher.add_geom(
        name="press_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0008, 0.004, 0.003],
        mass=0.03,
        rgba=[0.2, 0.8, 0.5, 1],
        solref=[0.001, 1],
        solimp=[0.99, 0.999, 0.00001, 0.5, 2],
    )
    for name, joint, kp, limit in (
        ("press", "press_slide", 1500.0, 4.0),
        ("pull", "plug_slide", 500.0, 3.0),
    ):
        actuator = spec.add_actuator(
            name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        actuator.set_to_position(kp=kp)
        actuator.forcelimited = True
        actuator.forcerange = [-limit, limit]
    spec.worldbody.add_camera(
        name="front", pos=[0.06, -0.22, 0.13], xyaxes=[1, 0, 0, 0, 0.45, 0.89]
    )
    return spec, names


class LatchFixture:
    def __init__(
        self,
        beam=None,
        latch=None,
        *,
        segments=8,
        timestep=0.0000025,
        press_x=None,
        press_y=0.0,
        max_force_n=5.0,
        max_deflection_m=0.012,
    ):
        self.beam, self.latch = beam or BeamParameters(), latch or LatchParameters()
        finite_number(max_force_n, "max_force_n")
        finite_number(max_deflection_m, "max_deflection_m")
        self.spec, self.names = build_latch_fixture(
            self.beam,
            self.latch,
            segments=segments,
            timestep=timestep,
            press_x=press_x,
            press_y=press_y,
        )
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.observer = LatchObserver(
            self.model,
            self.names,
            max_force_n=max_force_n,
            max_deflection_m=max_deflection_m,
        )
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.observer.reset()
        self.peak_press_force_n = 0.0
        self.peak_pull_force_n = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observer.observe(self.data)

    def step(self, press_position_m, pull_position_m, seconds=0.01):
        if self.observer.invalid_reason:
            raise RuntimeError("Invalid fixture must reset before continuing")
        values = np.asarray([press_position_m, pull_position_m, seconds])
        if not np.isfinite(values).all() or seconds <= 0:
            raise ValueError("Fixture commands must be finite with positive duration")
        ratio = seconds / self.model.opt.timestep
        if not 1 <= ratio <= 1_000_000 or not np.isclose(
            ratio, round(ratio), rtol=0, atol=1e-7
        ):
            raise ValueError(
                "Duration must be an integral bounded number of physics substeps"
            )
        n = round(ratio)
        self.data.ctrl[self.model.actuator("press").id] = press_position_m
        self.data.ctrl[self.model.actuator("pull").id] = pull_position_m
        press_id, pull_id = (
            self.model.actuator("press").id,
            self.model.actuator("pull").id,
        )
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            # 几何/接触需来自同一个当前状态，非积分前的旧 xpos。
            mujoco.mj_forward(self.model, self.data)
            self.peak_press_force_n = max(
                self.peak_press_force_n, abs(float(self.data.actuator_force[press_id]))
            )
            self.peak_pull_force_n = max(
                self.peak_pull_force_n, abs(float(self.data.actuator_force[pull_id]))
            )
            state = self.observer.observe(self.data)
            if state["invalid_reason"]:
                break
        state["press_force_n"] = float(
            self.data.actuator_force[self.model.actuator("press").id]
        )
        state["pull_force_n"] = float(
            self.data.actuator_force[self.model.actuator("pull").id]
        )
        state["pull_position_m"] = float(self.data.joint("plug_slide").qpos[0])
        state["peak_press_force_n"] = self.peak_press_force_n
        state["peak_pull_force_n"] = self.peak_pull_force_n
        return state


def run_fixture(
    case="press_pull", *, segments=8, timestep=0.0000025, beam=None, latch=None
):
    if case not in {
        "no_press",
        "partial_press",
        "offset_press",
        "press_pull",
        "early_release",
        "overload",
    }:
        raise ValueError("Unknown counterfactual")
    sim = LatchFixture(
        beam,
        latch,
        segments=segments,
        timestep=timestep,
        press_y=0.03 if case == "offset_press" else 0.0,
    )
    rows = []
    for k in range(300):
        t = k * 0.01
        press = -0.020 * min(1, t / 0.5)
        if case == "no_press":
            press = 0.0
        elif case == "partial_press":
            press = -0.0145 * min(1, t / 0.5)
        elif case == "overload":
            press = -0.040 * min(1, t / 0.5)
            # 明确的故障夹具外载，非真实任务动作：额外 10 N 向下压力。
            sim.data.xfrc_applied[sim.model.body("press_tool").id, 2] = -10 * min(
                1, t / 0.5
            )
        elif case == "early_release" and t > 1.15:
            # 已开始拉出、尚未越过肩部时平滑松开，验证真正的复锁而非只测试不按。
            press *= max(0.0, 1 - (t - 1.15) / 0.15)
        pull = -0.025 * np.clip((t - 1.0) / 1.8, 0, 1)
        state = sim.step(press, pull)
        rows.append(state)
        if state["invalid_reason"] or state["released"]:
            break
    return {
        "case": case,
        "parameters": {"beam": asdict(sim.beam), "latch": asdict(sim.latch)},
        "segments": segments,
        "timestep_s": timestep,
        "final": rows[-1],
        "events": sim.observer.events,
        "trace": rows,
    }
