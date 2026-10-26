"""长任务重新抓持：任意裸线段的双指垫接触与两手抓点分离。"""

from collections import deque
from types import SimpleNamespace

import numpy as np

from ibero.envs.cable_routing import CableRoutingEnv


def _contact_stub(left_element: int, right_element: int):
    env = object.__new__(CableRoutingEnv)
    points = np.column_stack((np.zeros(17), np.arange(17) * 0.018, np.zeros(17)))
    pinches = np.array([
        (points[left_element] + points[left_element + 1]) / 2,
        (points[right_element] + points[right_element + 1]) / 2,
    ])
    env.compiled = SimpleNamespace(cable=SimpleNamespace(segments=16))
    env.handles = SimpleNamespace(left_anchor_site_id=0, right_anchor_site_id=1)
    env.data = SimpleNamespace(
        site_xpos=pinches,
        site_xmat=np.repeat(np.eye(3).reshape(1, 9), 2, axis=0),
    )
    env._points = lambda: points
    env._flex_pad_contacts = lambda side: (
        True, {left_element if side == "left" else right_element}
    )
    env._grasp_history = {side: deque(maxlen=3) for side in ("left", "right")}
    return env


def test_two_distinct_midspan_regrasps_are_observed():
    # 旧观察器只接受左端/右端各 1/3，右手抓第10段会被误报未抓。
    env = _contact_stub(5, 10)
    for _ in range(3):
        env._update_grasps()
    assert env._grasped("left") and env._grasped("right")
    assert env._grasp_elements == {"left": 5, "right": 10}


def test_same_material_segment_cannot_count_as_two_grasps():
    env = _contact_stub(6, 6)
    for _ in range(3):
        env._update_grasps()
    assert not env._grasped("left") and not env._grasped("right")
    assert env._grasp_elements == {"left": None, "right": None}
