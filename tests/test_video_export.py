"""视频按仿真时间采样；测试不启动 GL 或仿真积分。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("export_demo_video", Path(__file__).resolve().parents[1] / "scripts/export_demo_video.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_normal_speed_preserves_start_and_end():
    times = np.linspace(3.0, 4.0, 101)
    result = module.frame_indices(times, 25, 1)
    assert len(result) == 26
    np.testing.assert_array_equal(result, np.arange(0, 101, 4))


def test_acceleration_changes_video_duration_not_states():
    times = np.linspace(0, 1, 101)
    result = module.frame_indices(times, 25, 2)
    assert len(result) == 14
    assert result[0] == 0 and result[-1] == 100
    assert np.all(np.diff(result) >= 0)


@pytest.mark.parametrize("times,fps,speed", [([], 25, 1), ([1, 0], 25, 1), ([0, np.nan], 25, 1), ([0, 1], 0, 1), ([0, 1], 25, 0), ([0, 1], 25, np.inf)])
def test_invalid_time_mapping_rejected(times, fps, speed):
    with pytest.raises(ValueError):
        module.frame_indices(times, fps, speed)


def test_single_frame_and_subframe_episode():
    np.testing.assert_array_equal(module.frame_indices([5], 25, 1), [0])
    np.testing.assert_array_equal(module.frame_indices([5, 5.001], 25, 1), [0, 1])


def test_success_fields_and_no_invented_success():
    assert module.case_passed({"is_success": True}) is True
    assert module.case_passed({"success": True}) is True
    assert module.case_passed({}, {"passed": True}) is True
    assert module.case_passed({"success": True}, {"passed": False}) is False
    assert module.case_passed({}) is None
