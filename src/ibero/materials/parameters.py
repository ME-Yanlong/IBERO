"""工业台架物性：SI 单位、显式来源；数值离散不伪装成材料参数。"""

from dataclasses import dataclass, fields
from numbers import Real
import math


def finite_number(value, name, *, minimum=0.0, positive=True):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be {'>' if positive else '>='} {minimum}")
    return float(value)


def strict_parameters(cls, values):
    """菜谱必须完整声明参数；Python 台架可使用 dataclass 的显式示例默认值。"""
    expected = {f.name for f in fields(cls)}
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError(f"{cls.__name__} requires exactly {sorted(expected)}")
    return cls(**values)


@dataclass(frozen=True)
class BeamParameters:
    length_m: float = 0.06
    width_m: float = 0.012
    thickness_m: float = 0.002
    young_pa: float = 2e9
    density_kg_m3: float = 1200.0
    # Kelvin–Voigt 弯曲松弛时间：关节阻尼 = 转角刚度 × 此时间。
    relaxation_time_s: float = 0.003
    parameter_source: str = "provisional rectangular polymer beam; not measured"

    def __post_init__(self):
        for field in fields(self):
            if field.name != "parameter_source":
                finite_number(getattr(self, field.name), field.name)
        if (
            not isinstance(self.parameter_source, str)
            or not self.parameter_source.strip()
        ):
            raise ValueError(
                "parameter_source must identify the reference or provisional assumption"
            )
        if self.thickness_m >= self.length_m / 5:
            raise ValueError(
                "Beam model requires a slender section: thickness < length/5"
            )
        try:
            for name in ("rigidity_nm2", "mass_kg", "tip_stiffness_n_m"):
                finite_number(getattr(self, name), name)
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError(
                "Beam derived quantities exceed numerical range"
            ) from error

    @property
    def rigidity_nm2(self):
        return self.young_pa * self.width_m * self.thickness_m**3 / 12

    @property
    def mass_kg(self):
        return self.length_m * self.width_m * self.thickness_m * self.density_kg_m3

    @property
    def tip_stiffness_n_m(self):
        return 3 * self.rigidity_nm2 / self.length_m**3


@dataclass(frozen=True)
class LatchParameters:
    hook_height_m: float = 0.006
    hook_length_m: float = 0.004
    overlap_m: float = 0.003
    axial_gap_m: float = 0.001
    friction: float = 0.3
    plug_mass_kg: float = 0.06

    def __post_init__(self):
        for field in fields(self):
            finite_number(
                getattr(self, field.name), field.name, positive=field.name != "friction"
            )
        if self.overlap_m >= self.hook_height_m:
            raise ValueError("overlap must be smaller than hook height")


@dataclass(frozen=True)
class StockParameters:
    size_m: tuple = (0.06, 0.04, 0.01)
    density_kg_m3: float = 7850.0
    material_grade: str = "AISI1045-provisional"
    parameter_source: str = "illustrative stock; no cutting calibration"

    def __post_init__(self):
        if not isinstance(self.size_m, (tuple, list)) or len(self.size_m) != 3:
            raise ValueError("size_m must have three positive dimensions")
        object.__setattr__(
            self, "size_m", tuple(finite_number(v, "size_m") for v in self.size_m)
        )
        finite_number(self.density_kg_m3, "density_kg_m3")
        for name in ("material_grade", "parameter_source"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} is required")
