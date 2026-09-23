"""
THUMBTYPE AI — finger/joint thumb-typing keyboard.

Pipeline per hand, per frame:
    thumb position -> nearest FINGER (index/middle/ring/pinky)
                    -> nearest JOINT on that finger (base/mid/tip)
                    -> letter, gated by a hold-time + lock/release
                       state machine so a touch has to be deliberate
                       before it registers as a keystroke.

This file is a hardened rewrite of the original: same algorithm and
feel, with the functional bugs and rough edges fixed (see inline
"FIX:" / "UPGRADE:" comments for what changed and why).
"""

import os
import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "hand_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Thumb must stay near target for this time
HOLD_TIME = 0.18

# Type cooldown
TYPE_COOLDOWN = 0.30

# Smoothing (0..1, higher = snappier / less smoothing)
SMOOTHING = 0.55

# Target radius relative to hand size
TARGET_RADIUS = 0.115

# Release radius
RELEASE_RADIUS = 0.155

# Finger selection margin
FINGER_MARGIN = 0.018

# Joint selection margin (was a bare 0.018 literal buried in
# find_joint() — pulled out into a named, independently-tunable
# constant like its finger-level counterpart above).
JOINT_MARGIN = 0.018

# MediaPipe's handedness label assumes the image it sees is already
# mirrored (selfie-style). This app flips the camera frame before
# detection, so labels normally need to be swapped back to match
# real left/right hands. If typing with your LEFT hand keeps
# producing RIGHT-hand letters (or vice versa) on your setup/camera,
# flip this one flag rather than hunting through the code.
SWAP_HANDEDNESS = True

# Below this confidence, don't trust a fresh Left/Right label for a
# given detection slot — keep whatever hand it was assigned to last
# frame instead. Cuts down on hand-identity flicker at frame edges
# or during fast motion.
HANDEDNESS_MIN_CONFIDENCE = 0.6

# How many consecutive frames a hand can go undetected before we
# treat it as "gone" and reset its smoothing/lock state. A grace
# period avoids punishing single-frame detector dropouts (common
# and harmless) while still cleaning up state when a hand actually
# leaves the frame.
GRACE_FRAMES = 5


# ============================================================
# LETTER MAPPING
# ============================================================

LEFT_KEYS = {
    8: "A",
    6: "B",
    5: "C",

    12: "D",
    10: "E",
    9: "F",

    16: "G",
    14: "H",
    13: "I",

    20: "J",
    18: "K",
    17: "L",
}

RIGHT_KEYS = {
    8: "M",
    6: "N",
    5: "O",

    12: "P",
    10: "Q",
    9: "R",

    16: "S",
    14: "T",
    13: "U",

    20: "V",
    18: "W",
    17: "X",
}


# ============================================================
# FINGER DEFINITIONS
# ============================================================

FINGERS = {
    "INDEX": {
        "base": 5,
        "middle": 6,
        "tip": 8,
        "targets": [5, 6, 8],
    },

    "MIDDLE": {
        "base": 9,
        "middle": 10,
        "tip": 12,
        "targets": [9, 10, 12],
    },

    "RING": {
        "base": 13,
        "middle": 14,
        "tip": 16,
        "targets": [13, 14, 16],
    },

    "PINKY": {
        "base": 17,
        "middle": 18,
        "tip": 20,
        "targets": [17, 18, 20],
    },
}


# ============================================================
# MEDIAPIPE
#
# UPGRADE: RunningMode.VIDEO instead of IMAGE. IMAGE mode re-solves
# hand pose from scratch every frame with zero temporal context.
# VIDEO mode is fed a monotonically increasing timestamp and lets
# MediaPipe use the previous frame to track the hand, which gives
# noticeably steadier landmarks on a live webcam feed.
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise SystemExit(
        f"❌ Model file not found: {MODEL_PATH}\n"
        "   Download hand_landmarker.task and place it next to this "
        "script before running."
    )

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,

    min_hand_detection_confidence=0.60,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.60
)

try:

    detector = vision.HandLandmarker.create_from_options(
        options
    )

except Exception as e:

    raise SystemExit(f"❌ Could not load hand landmarker model: {e}")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

# FIX: original script had no open-check at all — a missing/busy
# camera would silently fail on the first frame read with no
# explanation.
if not cap.isOpened():

    raise SystemExit("❌ Camera could not be opened.")


# ============================================================
# GLOBAL TEXT
# ============================================================

typed_text = ""


# ============================================================
# HAND STATE
# ============================================================

def make_default_state():

    return {
        "candidate": None,
        "candidate_since": 0,
        "locked": False,
        "locked_key": None,
        "locked_joint": None,   # FIX: needed to detect key-to-key slides
        "last_type": 0,
        "distance": 999,
        "finger": None,
        "joint": None,
    }


states = {
    "Left": make_default_state(),
    "Right": make_default_state(),
}


# ============================================================
# SMOOTHING STORAGE
# ============================================================

previous = {
    "Left": None,
    "Right": None
}

# How many consecutive frames each hand has been missing.
missed_frames = {
    "Left": 0,
    "Right": 0,
}

# Last confidently-assigned hand name per detection slot index,
# used to smooth over low-confidence handedness flicker.
last_hand_by_slot = {}


def reset_hand(hand_name):
    """
    Fully clear a hand's smoothing history and typing state. Called
    once a hand has been missing for GRACE_FRAMES in a row, so a
    hand that briefly drops out for a frame or two doesn't lose its
    smoothing baseline or an in-progress hold for no reason, but a
    hand that actually leaves the frame doesn't leave stale state
    behind (which previously could cause a "ghost" locked letter or
    a smoothing glide-artifact when a hand reappeared).
    """

    previous[hand_name] = None
    states[hand_name] = make_default_state()


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def dist2d(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def clamp(v, a, b):

    return max(a, min(b, v))


def lerp(a, b, amount):

    return a + (b - a) * amount


# ============================================================
# LANDMARK POINT
#
# FIX: the original smoothing step rebuilt landmarks with
# `type(p)(x=x, y=y, z=z)`, silently relying on the exact
# constructor signature of MediaPipe's internal landmark class.
# A small dependency-version change could break that with a
# confusing error deep inside the render loop. A tiny local Point
# class is explicit, faster to construct, and immune to upstream
# API changes.
# ============================================================

class Point:

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# ============================================================
# LANDMARK SMOOTHING
# ============================================================

def smooth_landmarks(hand_name, landmarks):

    old = previous[hand_name]

    if old is None or len(old) != len(landmarks):

        result = [
            Point(p.x, p.y, p.z)
            for p in landmarks
        ]

        previous[hand_name] = result

        return result

    result = []

    for p, o in zip(landmarks, old):

        x = lerp(o.x, p.x, SMOOTHING)
        y = lerp(o.y, p.y, SMOOTHING)
        z = lerp(o.z, p.z, SMOOTHING)

        result.append(Point(x, y, z))

    previous[hand_name] = result

    return result


# ============================================================
# HAND SCALE
# ============================================================

def get_hand_scale(points):

    wrist = points[0]

    middle_base = points[9]

    scale = dist2d(
        wrist,
        middle_base
    )

    if scale < 0.05:
        scale = 0.05

    return scale


# ============================================================
# POINT TO LINE SEGMENT DISTANCE
# ============================================================

def point_segment_distance(
    px,
    py,
    ax,
    ay,
    bx,
    by
):

    abx = bx - ax
    aby = by - ay

    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby

    if ab2 == 0:

        return math.sqrt(
            (px - ax) ** 2 +
            (py - ay) ** 2
        )

    t = (
        apx * abx +
        apy * aby
    ) / ab2

    t = clamp(
        t,
        0.0,
        1.0
    )

    cx = ax + t * abx
    cy = ay + t * aby

    return math.sqrt(
        (px - cx) ** 2 +
        (py - cy) ** 2
    )


# ============================================================
# FIND WHICH FINGER THUMB IS POINTING AT
# ============================================================

def find_finger(
    thumb,
    landmarks,
    scale
):

    candidates = []

    for finger_name, info in FINGERS.items():

        base = landmarks[info["base"]]
        middle = landmarks[info["middle"]]
        tip = landmarks[info["tip"]]

        d1 = point_segment_distance(
            thumb.x,
            thumb.y,
            base.x,
            base.y,
            middle.x,
            middle.y
        )

        d2 = point_segment_distance(
            thumb.x,
            thumb.y,
            middle.x,
            middle.y,
            tip.x,
            tip.y
        )

        d = min(d1, d2)

        normalized = d / scale

        candidates.append(
            (normalized, finger_name)
        )

    candidates.sort(key=lambda x: x[0])

    best_distance = candidates[0][0]
    best_finger = candidates[0][1]

    second_distance = candidates[1][0]

    if second_distance - best_distance < FINGER_MARGIN:
        return None, best_distance

    return best_finger, best_distance


# ============================================================
# FIND TARGET JOINT
# ============================================================

def find_joint(
    thumb,
    landmarks,
    finger_name,
    scale
):

    info = FINGERS[finger_name]

    targets = info["targets"]

    values = []

    for joint_id in targets:

        d = dist2d(
            thumb,
            landmarks[joint_id]
        )

        normalized = d / scale

        values.append((normalized, joint_id))

    values.sort(key=lambda x: x[0])

    best_distance, best_joint = values[0]

    second_distance = values[1][0]

    # FIX: this ambiguity check used a bare 0.018 literal instead of
    # the named JOINT_MARGIN constant defined at the top of the file.
    if second_distance - best_distance < JOINT_MARGIN:
        return None, best_distance

    return best_joint, best_distance


# ============================================================
# GET LETTER
# ============================================================

def get_letter(hand_name, joint):

    if hand_name == "Left":
        return LEFT_KEYS.get(joint)

    return RIGHT_KEYS.get(joint)


# ============================================================
# PROCESS HAND
# ============================================================

def process_hand(hand_name, landmarks):

    state = states[hand_name]

    now = time.time()

    thumb = landmarks[4]

    scale = get_hand_scale(landmarks)

    # --------------------------------------------------------
    # STEP 1 — which finger
    # --------------------------------------------------------

    finger, finger_distance = find_finger(
        thumb,
        landmarks,
        scale
    )

    state["finger"] = finger

    if finger is None:

        state["candidate"] = None
        state["candidate_since"] = 0

        if not state["locked"]:
            state["joint"] = None

        return None

    # --------------------------------------------------------
    # STEP 2 — which joint on that finger
    # --------------------------------------------------------

    joint, joint_distance = find_joint(
        thumb,
        landmarks,
        finger,
        scale
    )

    state["distance"] = joint_distance

    if joint is None:

        state["candidate"] = None
        state["candidate_since"] = 0

        return None

    target_radius = TARGET_RADIUS
    release_radius = RELEASE_RADIUS

    # --------------------------------------------------------
    # Outside target
    # --------------------------------------------------------

    if joint_distance > target_radius:

        if state["locked"]:

            if joint_distance > release_radius:

                state["locked"] = False
                state["locked_key"] = None
                state["locked_joint"] = None
                state["candidate"] = None
                state["candidate_since"] = 0

        else:

            state["candidate"] = None
            state["candidate_since"] = 0

        return None

    # --------------------------------------------------------
    # Letter
    # --------------------------------------------------------

    letter = get_letter(hand_name, joint)

    if letter is None:
        return None

    state["joint"] = joint

    # --------------------------------------------------------
    # Already locked on a key
    # --------------------------------------------------------

    if state["locked"]:

        if joint == state["locked_joint"]:

            # Still parked on the same key — nothing new to do.
            return state["locked_key"]

        # FIX (the "stuck lock" bug): previously, once `locked` was
        # True, this branch always returned `locked_key` no matter
        # what the thumb was actually touching now. If you slid the
        # thumb directly from one key to an adjacent one without
        # ever crossing RELEASE_RADIUS in between (very easy to do,
        # since target zones for neighbouring joints can be close
        # together), the lock never released and every letter after
        # the first one was silently swallowed.
        #
        # Now: landing on a *different* joint while still locked
        # releases the old lock immediately and falls through to
        # the normal candidate/hold logic below, so the new key
        # still needs its own deliberate hold — it isn't typed
        # instantly — but it's no longer ignored.
        state["locked"] = False
        state["locked_key"] = None
        state["locked_joint"] = None
        state["candidate"] = None
        state["candidate_since"] = 0

    # --------------------------------------------------------
    # New candidate
    # --------------------------------------------------------

    if state["candidate"] != letter:

        state["candidate"] = letter
        state["candidate_since"] = now

        return letter

    # --------------------------------------------------------
    # Stable hold
    # --------------------------------------------------------

    elapsed = now - state["candidate_since"]

    if elapsed >= HOLD_TIME:

        if now - state["last_type"] >= TYPE_COOLDOWN:

            type_letter(letter)

            state["last_type"] = now
            state["locked"] = True
            state["locked_key"] = letter
            state["locked_joint"] = joint

    return letter


# ============================================================
# TYPE
# ============================================================

def type_letter(letter):

    global typed_text

    typed_text += letter


# ============================================================
# DRAW SKELETON
# ============================================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_skeleton(frame, landmarks):

    h, w = frame.shape[:2]

    for a, b in CONNECTIONS:

        p1 = landmarks[a]
        p2 = landmarks[b]

        x1 = int(p1.x * w)
        y1 = int(p1.y * h)

        x2 = int(p2.x * w)
        y2 = int(p2.y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (100, 220, 255),
            2
        )

    for i, p in enumerate(landmarks):

        x = int(p.x * w)
        y = int(p.y * h)

        radius = 11 if i == 4 else 5

        cv2.circle(
            frame,
            (x, y),
            radius,
            (255, 255, 255),
            -1
        )


# ============================================================
# DRAW LETTERS
# ============================================================

def draw_letters(frame, landmarks, hand_name, active_joint):

    h, w = frame.shape[:2]

    mapping = LEFT_KEYS if hand_name == "Left" else RIGHT_KEYS

    for finger_name, info in FINGERS.items():

        for joint_id in info["targets"]:

            p = landmarks[joint_id]

            x = int(p.x * w)
            y = int(p.y * h)

            letter = mapping[joint_id]

            radius = 25

            if joint_id == active_joint:

                cv2.circle(frame, (x, y), 34, (0, 255, 0), 3)
                cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)

                text_color = (0, 255, 0)

            else:

                cv2.circle(frame, (x, y), radius, (70, 70, 70), 1)

                text_color = (0, 255, 255)

            cv2.putText(
                frame,
                letter,
                (x + 12, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                text_color,
                2,
                cv2.LINE_AA
            )


# ============================================================
# DRAW THUMB
# ============================================================

def draw_thumb(frame, landmarks, hand_name):

    h, w = frame.shape[:2]

    thumb = landmarks[4]

    x = int(thumb.x * w)
    y = int(thumb.y * h)

    state = states[hand_name]

    cv2.circle(frame, (x, y), 15, (0, 255, 0), 3)

    cv2.putText(
        frame,
        "THUMB",
        (x + 18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    if state["candidate"]:

        cv2.putText(
            frame,
            state["candidate"],
            (x + 18, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )


# ============================================================
# DRAW TEXT BOX
# ============================================================

def draw_text_box(frame):

    h, w = frame.shape[:2]

    cv2.rectangle(frame, (25, 20), (w - 25, 105), (15, 15, 15), -1)
    cv2.rectangle(frame, (25, 20), (w - 25, 105), (90, 90, 90), 2)

    display = typed_text

    if len(display) > 80:
        display = display[-80:]

    cv2.putText(
        frame,
        display,
        (45, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# STATUS PANEL
# ============================================================

def draw_status(frame, left_detected, right_detected, fps):

    x = 25
    y = 130

    width = 270
    height = 230

    cv2.rectangle(frame, (x, y), (x + width, y + height), (15, 15, 15), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 2)

    cv2.putText(
        frame,
        "THUMB TYPE",
        (x + 15, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    left_state = states["Left"]
    right_state = states["Right"]

    cv2.putText(
        frame, "LEFT", (x + 15, y + 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
    )

    cv2.putText(
        frame, str(left_state["candidate"] or "-"), (x + 95, y + 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )

    cv2.putText(
        frame, "RIGHT", (x + 15, y + 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
    )

    cv2.putText(
        frame, str(right_state["candidate"] or "-"), (x + 95, y + 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )

    cv2.putText(
        frame, "Left:  A - L", (x + 15, y + 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1
    )

    cv2.putText(
        frame, "Right: M - X", (x + 15, y + 160),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1
    )

    cv2.putText(
        frame, "C=CLEAR  B=BACKSPACE  SPACE  Q=EXIT", (x + 15, y + 188),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1
    )

    cv2.putText(
        frame, f"FPS: {fps:4.1f}", (x + 15, y + 215),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1
    )


# ============================================================
# MAIN LOOP
# ============================================================

# Monotonic timestamp counter, required by VIDEO mode.
last_timestamp_ms = 0

fps = 0.0

try:

    while True:

        loop_start = time.time()

        success, frame = cap.read()

        if not success:
            break

        # Mirror
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.time() * 1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        detected_hands = set()

        # ====================================================
        # HANDS
        # ====================================================

        if result.hand_landmarks:

            for i, raw_landmarks in enumerate(result.hand_landmarks):

                if i >= len(result.handedness):
                    continue

                category = result.handedness[i][0]

                raw_label = category.category_name
                score = category.score

                if SWAP_HANDEDNESS:
                    proposed_name = "Right" if raw_label == "Left" else "Left"
                else:
                    proposed_name = raw_label

                # FIX: low-confidence handedness used to be trusted
                # outright, which could momentarily swap which
                # hand's letters you were typing. Below the
                # confidence threshold, keep the previous frame's
                # assignment for this detection slot instead.
                if score >= HANDEDNESS_MIN_CONFIDENCE:

                    last_hand_by_slot[i] = proposed_name
                    hand_name = proposed_name

                else:

                    hand_name = last_hand_by_slot.get(i, proposed_name)

                detected_hands.add(hand_name)

                landmarks = smooth_landmarks(hand_name, raw_landmarks)

                draw_skeleton(frame, landmarks)

                process_hand(hand_name, landmarks)

                state = states[hand_name]

                draw_letters(frame, landmarks, hand_name, state["joint"])

                draw_thumb(frame, landmarks, hand_name)

        # ====================================================
        # GRACE-PERIOD RESET FOR MISSING HANDS
        #
        # FIX: a hand's smoothing history and lock state used to
        # persist forever once set, even after the hand left the
        # frame. If it reappeared later (or a different hand got
        # assigned to the same slot), stale smoothing data could
        # cause a brief "glide" through old positions, and a stale
        # `locked` flag could make a freshly-raised hand appear to
        # already be holding a key. Now each hand gets a short grace
        # window before its state is cleared.
        # ====================================================

        for hand_name in ("Left", "Right"):

            if hand_name in detected_hands:

                missed_frames[hand_name] = 0

            else:

                missed_frames[hand_name] += 1

                if missed_frames[hand_name] == GRACE_FRAMES:
                    reset_hand(hand_name)

        # ====================================================
        # FPS
        # ====================================================

        loop_end = time.time()

        elapsed = loop_end - loop_start

        if elapsed > 0:

            instant_fps = 1.0 / elapsed

            fps = lerp(fps, instant_fps, 0.1)

        # ====================================================
        # STATUS
        # ====================================================

        draw_text_box(frame)

        draw_status(
            frame,
            "Left" in detected_hands,
            "Right" in detected_hands,
            fps
        )

        # ====================================================
        # INSTRUCTIONS (centered, was a fixed x=320 before)
        # ====================================================

        instructions = "THUMB -> JOINT -> HOLD -> RELEASE"

        (text_w, _), _ = cv2.getTextSize(
            instructions,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        frame_h, frame_w = frame.shape[:2]

        cv2.putText(
            frame,
            instructions,
            ((frame_w - text_w) // 2, frame_h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow("THUMBTYPE AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("c"):

            typed_text = ""

            for hand in states:
                states[hand] = make_default_state()

        elif key == ord("b"):

            # New: backspace. Previously the only way to correct a
            # mistake was to wipe the entire line with 'c'.
            typed_text = typed_text[:-1]

        elif key == 32:  # spacebar

            # New: space, so words can actually be separated without
            # a hand gesture dedicated to it.
            typed_text += " "

finally:

    # ============================================================
    # CLEANUP
    #
    # FIX: originally not wrapped in try/finally, so an exception
    # mid-loop (or Ctrl+C) could leave the camera locked and the
    # OpenCV window open.
    # ============================================================

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    print()
    print("==========================================")
    print("FINAL TEXT")
    print("==========================================")
    print(typed_text)