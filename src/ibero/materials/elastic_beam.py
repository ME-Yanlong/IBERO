"""分段被动悬臂梁：转角弹簧近似 EI 曲率能，不由任务指定弯曲动画。"""

from dataclasses import dataclass
import mujoco

from ibero.materials.parameters import BeamParameters


@dataclass(frozen=True)
class BeamNames:
    joints: tuple[str, ...]
    bodies: tuple[str, ...]
    tip_site: str


def add_elastic_beam(parent, params: BeamParameters, segments=8, prefix="tongue_"):
    """平面弯曲模型，固定根部平移。第一个半单元刚度为 2EI/h。

    根部使用半单元积分，可使静态端点柔度误差随 h² 收敛；其余关节
    k=EI/h，d=k*tau。材料阻尼 tau 是明示的等效参数，尚未实测标定。
    """
    if type(segments) is not int or not 4 <= segments <= 32:
        raise ValueError("segments must be an integer in [4, 32]")
    h = params.length_m / segments
    joints, bodies = [], []
    for i in range(segments):
        body = parent.add_body(
            name=f"{prefix}segment_{i}", pos=[0 if i == 0 else h, 0, 0]
        )
        stiffness = params.rigidity_nm2 / h * (2 if i == 0 else 1)
        joint = body.add_joint(
            name=f"{prefix}bend_{i}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            stiffness=stiffness,
            damping=stiffness * params.relaxation_time_s,
        )
        body.add_geom(
            name=f"{prefix}beam_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[h / 2, 0, 0],
            size=[h / 2, params.width_m / 2, params.thickness_m / 2],
            mass=params.mass_kg / segments,
            rgba=[0.9, 0.6, 0.12, 1],
            friction=[0.3, 0.001, 0.0001],
            solref=[0.001, 1],
            solimp=[0.99, 0.999, 0.00001, 0.5, 2],
        )
        joints.append(joint.name)
        bodies.append(body.name)
        parent = body
    tip = parent.add_site(
        name=f"{prefix}tip", pos=[h, 0, 0], size=[0.001], rgba=[1, 0, 0, 1]
    )
    return parent, BeamNames(tuple(joints), tuple(bodies), tip.name)
