"""카메라 프로파일 감시 — Studio 가 기본값으로 되돌린 값만 프로파일 값으로 다시 넣는다."""

from tools.camera_guard import plan_fixes

CAM = {"id": "rs:1:color", "profile_key": "rs:1:color", "connected": True}
ENTRY = {"key": "rs:1:color", "match": {"last_dev": "rs:1:color"},
         "controls": {"exposure": 56, "gain": 64, "white_balance": 4380,
                      "enable_auto_exposure": 0, "enable_auto_white_balance": 0}}


def controls(exposure=166, wb=4380, ae=0):
    return lambda _id: {"exposure": {"value": exposure, "default": 166},
                        "white_balance": {"value": wb, "default": 4600},
                        "gain": {"value": 64, "default": 64},
                        "enable_auto_exposure": {"value": ae, "default": 1},
                        "enable_auto_white_balance": {"value": 0, "default": 1}}


def test_exposure_reset_to_default_is_restored():
    assert plan_fixes([CAM], [ENTRY], controls()) == [("rs:1:color", "exposure", 56.0)]


def test_values_set_by_hand_or_gray_card_are_left_alone():
    assert plan_fixes([CAM], [ENTRY], controls(exposure=72)) == []        # 보정 중·사람이 바꾼 값
    assert plan_fixes([CAM], [ENTRY], controls(exposure=56)) == []        # 이미 맞다


def test_auto_exposure_turned_back_on_is_switched_off_first():
    assert plan_fixes([CAM], [ENTRY], controls(ae=1)) == [("rs:1:color", "enable_auto_exposure", 0.0),
                                                          ("rs:1:color", "exposure", 56.0)]


def test_white_balance_reset_and_profile_equal_to_default():
    assert plan_fixes([CAM], [ENTRY], controls(exposure=56, wb=4600)) == [("rs:1:color", "white_balance", 4380.0)]
    gain_default = {**ENTRY, "controls": {"gain": 64}}
    assert plan_fixes([CAM], [gain_default], controls()) == []           # 프로파일도 기본값이면 할 일 없음


def test_disconnected_or_unmatched_cameras_are_skipped():
    assert plan_fixes([{**CAM, "connected": False}], [ENTRY], controls()) == []
    assert plan_fixes([{**CAM, "profile_key": "x", "id": "y"}], [ENTRY], controls()) == []
    by_last_dev = {**CAM, "profile_key": "usb:4-2"}                      # 키가 달라도 last_dev 로 짝짓는다
    assert plan_fixes([by_last_dev], [ENTRY], controls()) == [("rs:1:color", "exposure", 56.0)]
