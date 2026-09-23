"""Recognition helpers on fake MediaPipe results: label table, conversion, strict timestamps."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gesture_remote.observation import NONE_LABEL, NUM_LANDMARKS, Handedness
from gesture_remote.recognition import (
    CANNED_LABEL_SET,
    CANNED_LABELS,
    LabelMapper,
    MediaPipeGestureRecognizer,
    StrictTimestamps,
    to_observation,
)

AGREED_LABELS = {
    "none",
    "closed_fist",
    "open_palm",
    "pointing_up",
    "thumb_down",
    "thumb_up",
    "victory",
    "i_love_you",
}
"""The set agreed with lot B (its tests hard-code these strings)."""

SIZE = (640, 480)


def category(name: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(category_name=name, score=score)


def landmarks(offset: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=offset + i, y=offset + 2 * i, z=-i) for i in range(NUM_LANDMARKS)]


def fake_result(*hands: tuple[list[SimpleNamespace], str, float]) -> SimpleNamespace:
    """hands: (gesture categories, handedness name, handedness score)."""
    return SimpleNamespace(
        gestures=[gestures for gestures, _, _ in hands],
        handedness=[[category(side, side_score)] for _, side, side_score in hands],
        hand_landmarks=[landmarks(0.0) for _ in hands],
        hand_world_landmarks=[landmarks(100.0) for _ in hands],
    )


def test_label_table_matches_the_agreed_set() -> None:
    assert set(CANNED_LABEL_SET) == AGREED_LABELS
    assert MediaPipeGestureRecognizer.labels == CANNED_LABEL_SET
    assert CANNED_LABELS["None"] == CANNED_LABELS["Unknown"] == NONE_LABEL


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("None", "none"),
        ("Unknown", "none"),
        ("Closed_Fist", "closed_fist"),
        ("Open_Palm", "open_palm"),
        ("Pointing_Up", "pointing_up"),
        ("Thumb_Down", "thumb_down"),
        ("Thumb_Up", "thumb_up"),
        ("Victory", "victory"),
        ("ILoveYou", "i_love_you"),
    ],
)
def test_every_raw_class_converts(raw: str, label: str) -> None:
    observation = to_observation(
        fake_result(([category(raw, 0.9)], "Right", 0.97)), SIZE, LabelMapper()
    )
    (hand,) = observation.hands
    assert hand.label == label


def test_unknown_class_warns_once_per_name(caplog: pytest.LogCaptureFixture) -> None:
    mapper = LabelMapper()
    with caplog.at_level(logging.WARNING, logger="gesture_remote.recognition"):
        results = [mapper(name) for name in ("Shaka", "Shaka", "Rock", "Shaka", "Thumb_Up", "Rock")]
    assert results == ["none", "none", "none", "none", "thumb_up", "none"]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 2
    assert "'Shaka'" in warnings[0] and "'Rock'" in warnings[1]


def test_conversion_keeps_scores_handedness_and_landmarks() -> None:
    result = fake_result(
        ([category("Victory", 0.91), category("None", 0.05)], "Right", 0.98),
        ([category("Thumb_Up", 0.73)], "Left", 0.88),
    )
    observation = to_observation(result, SIZE, LabelMapper())
    assert isinstance(observation.hands, tuple)
    assert observation.image_size == SIZE
    first, second = observation.hands
    assert (first.label, first.score, first.handedness) == (
        "victory",
        pytest.approx(0.91),
        Handedness.RIGHT,
    )
    assert first.handedness_score == pytest.approx(0.98)
    assert (second.label, second.handedness) == ("thumb_up", Handedness.LEFT)
    assert first.landmarks.shape == first.world_landmarks.shape == (NUM_LANDMARKS, 3)
    assert first.landmarks.dtype == np.float32
    np.testing.assert_allclose(first.landmarks[3], (3.0, 6.0, -3.0))
    np.testing.assert_allclose(first.world_landmarks[3], (103.0, 106.0, -3.0))
    assert observation.primary() is first


def test_no_hand_gives_an_empty_tuple() -> None:
    observation = to_observation(fake_result(), SIZE, LabelMapper())
    assert observation.hands == ()
    assert observation.primary() is None


def test_hand_with_an_empty_gesture_list_is_none_with_zero_score() -> None:
    observation = to_observation(fake_result(([], "Left", 0.9)), SIZE, LabelMapper())
    (hand,) = observation.hands
    assert (hand.label, hand.score, hand.handedness) == (NONE_LABEL, 0.0, Handedness.LEFT)


def test_wrong_landmark_count_raises() -> None:
    result = fake_result(([category("Victory", 0.9)], "Right", 0.9))
    result.hand_landmarks[0] = result.hand_landmarks[0][:20]
    with pytest.raises(ValueError, match="21 landmarks"):
        to_observation(result, SIZE, LabelMapper())


def test_strict_timestamps_pass_increasing_values_through() -> None:
    strict = StrictTimestamps()
    assert [strict(t) for t in (0, 33, 66, 1000)] == [0, 33, 66, 1000]


def test_strict_timestamps_fix_a_stalled_clock() -> None:
    strict = StrictTimestamps()
    assert [strict(t) for t in (500, 500, 500, 501, 540)] == [500, 501, 502, 503, 540]


def test_strict_timestamps_fix_a_clock_going_back() -> None:
    strict = StrictTimestamps()
    assert [strict(t) for t in (1000, 990, 0, 1001, 1500)] == [1000, 1001, 1002, 1003, 1500]
