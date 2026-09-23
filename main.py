import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# THUMPTYPE
# ============================================================

MODEL_PATH = "hand_landmarker.task"

# Reference tuning
ZONE_SIZE = 0.16
CLEARANCE = 0.45
MARGIN = 0.35
RELEASE = 1.70
DWELL = 3
COOLDOWN = 0.20
SMOOTH = 0.55
PAD = 0.22

depth_check = True
swap_hands = False

typed_text = ""


# ============================================================
# KEY MAP
# ============================================================

TARGETS = [
    (8,  "Index",  "tip"),
    (6,  "Index",  "middle"),
    (5,  "Index",  "base"),

    (12, "Middle", "tip"),
    (10, "Middle", "middle"),
    (9,  "Middle", "base"),

    (16, "Ring",   "tip"),
    (14, "Ring",   "middle"),
    (13, "Ring",   "base"),

    (20, "Little", "tip"),
    (18, "Little", "middle"),
    (17, "Little", "base"),
]

# IMPORTANT:
# This follows the uploaded reference:
# RIGHT = A-L
# LEFT  = M-X

LETTERS = {
    "right": list("ABCDEFGHIJKL"),
    "left":  list("MNOPQRSTUVWX")
}

PAIRS = [
    (4,  "Y",     ("char", "Y"),     True),
    (8,  "Z",     ("char", "Z"),     False),
    (12, "ENTER", ("enter", None),   False),
    (20, "DELETE",("back", None),    False),
]


# ============================================================
# HAND CONNECTIONS
# ============================================================

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# ============================================================
# HAND STATE
# ============================================================

def new_hand():
    return {
        "seen": False,
        "was": False,
        "pts": None,
        "smooth": None,
        "scale": 1.0,
        "rad": [],
        "stylus": None,

        "near": -1,
        "score": 9.0,
        "gap": 9.0,

        "held": 0,
        "held_key": -1,

        "armed": True,
        "last": 0.0,

        "flash": 0.0,
        "flash_key": -1,

        "key": ""
    }


hands = {
    "left": new_hand(),
    "right": new_hand()
}


pair_state = {
    "Y": {"armed": True, "last": 0},
    "Z": {"armed": True, "last": 0},
    "ENTER": {"armed": True, "last": 0},
    "DELETE": {"armed": True, "last": 0},
    "SPACE": {"armed": True, "last": 0},
}


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def distance(a, b):

    dz = 0

    if depth_check:
        dz = (a["z"] - b["z"]) * 0.5

    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        dz ** 2
    )


def lerp(a, b, k):

    return {
        "x": a["x"] + (b["x"] - a["x"]) * k,
        "y": a["y"] + (b["y"] - a["y"]) * k,
        "z": a["z"] + (b["z"] - a["z"]) * k
    }


def palm_center(points):

    ids = [0, 5, 9, 13, 17]

    return {
        "x": sum(points[i]["x"] for i in ids) / 5,
        "y": sum(points[i]["y"] for i in ids) / 5,
        "z": 0
    }


# ============================================================
# OUTPUT
# ============================================================

def emit(kind, value=None):

    global typed_text

    if kind == "char":
        typed_text += value

    elif kind == "space":
        typed_text += " "

    elif kind == "enter":
        typed_text += "\n"

    elif kind == "back":
        typed_text = typed_text[:-1]


# ============================================================
# HAND MEASUREMENT
# ============================================================

def measure_hand(hand, raw):

    hand["seen"] = True

    # --------------------------------------------
    # Smooth landmarks
    # --------------------------------------------

    if hand["smooth"] is None or not hand["was"]:

        hand["smooth"] = [
            {
                "x": p["x"],
                "y": p["y"],
                "z": p["z"]
            }
            for p in raw
        ]

    else:

        for i, p in enumerate(raw):

            s = hand["smooth"][i]

            s["x"] += (p["x"] - s["x"]) * SMOOTH
            s["y"] += (p["y"] - s["y"]) * SMOOTH
            s["z"] += (p["z"] - s["z"]) * SMOOTH


    pts = hand["smooth"]
    hand["pts"] = pts


    # --------------------------------------------
    # Hand scale
    # --------------------------------------------

    hand["scale"] = max(
        math.hypot(
            pts[0]["x"] - pts[9]["x"],
            pts[0]["y"] - pts[9]["y"]
        ),
        0.000001
    )


    # --------------------------------------------
    # Thumb PAD / stylus
    # IMPORTANT
    # --------------------------------------------

    hand["stylus"] = lerp(
        pts[4],
        pts[3],
        PAD
    )


    # --------------------------------------------
    # Target positions
    # --------------------------------------------

    targets = []

    for lm, _, _ in TARGETS:
        targets.append(pts[lm])


    # --------------------------------------------
    # Dynamic non-overlapping zones
    # --------------------------------------------

    radii = []

    for i, p in enumerate(targets):

        nearest = 9

        for j, q in enumerate(targets):

            if i == j:
                continue

            d = math.hypot(
                p["x"] - q["x"],
                p["y"] - q["y"]
            ) / hand["scale"]

            nearest = min(nearest, d)

        radius = max(
            min(ZONE_SIZE, nearest * CLEARANCE),
            0.045
        )

        radii.append(radius)

    hand["rad"] = radii


    # --------------------------------------------
    # Find nearest target
    # --------------------------------------------

    best = -1
    score1 = 99
    score2 = 99

    for i, p in enumerate(targets):

        score = (
            distance(hand["stylus"], p)
            / hand["scale"]
        ) / radii[i]

        if score < score1:

            score2 = score1
            score1 = score
            best = i

        elif score < score2:

            score2 = score


    hand["near"] = best
    hand["score"] = score1
    hand["gap"] = score2 - score1


    # Thumb moved away -> re-arm
    if score1 > RELEASE:
        hand["armed"] = True


# ============================================================
# TYPE LETTER
# ============================================================

def commit_hand(hand, now, blocked):

    if blocked:

        hand["held"] = 0
        hand["held_key"] = -1
        hand["armed"] = False

        return


    # Thumb must be inside one clear zone
    if hand["score"] < 1 and hand["gap"] > MARGIN:

        # New key
        if hand["near"] != hand["held_key"]:

            hand["held_key"] = hand["near"]
            hand["held"] = 0


        hand["held"] += 1


        # Dwell + cooldown + armed
        if (
            hand["held"] >= DWELL
            and hand["armed"]
            and now - hand["last"] > COOLDOWN
        ):

            hand["armed"] = False
            hand["last"] = now

            hand["flash"] = now
            hand["flash_key"] = hand["near"]

            letter = LETTERS[
                hand["key"]
            ][hand["near"]]

            emit("char", letter)


    else:

        hand["held"] = 0
        hand["held_key"] = -1


# ============================================================
# TWO HAND GESTURES
# ============================================================

def fire_pair(name, kind, value, now):

    state = pair_state[name]

    if not state["armed"]:
        return False

    if now - state["last"] < COOLDOWN * 1.6:
        return False

    state["armed"] = False
    state["last"] = now

    emit(kind, value)

    return True


def read_pairs(now):

    left = hands["left"]
    right = hands["right"]


    if not (left["seen"] and right["seen"]):

        for state in pair_state.values():
            state["armed"] = True

        return None


    scale = (
        left["scale"] +
        right["scale"]
    ) / 2


    # --------------------------------------------
    # PALMS -> SPACE
    # --------------------------------------------

    lp = palm_center(left["pts"])
    rp = palm_center(right["pts"])

    palms = math.hypot(
        lp["x"] - rp["x"],
        lp["y"] - rp["y"]
    ) / scale


    if palms > 1.9:
        pair_state["SPACE"]["armed"] = True


    if palms < 0.90:

        fire_pair(
            "SPACE",
            "space",
            None,
            now
        )

        return "SPACE"


    # --------------------------------------------
    # Fingertip pairs
    # --------------------------------------------

    distances = []

    for lm, _, _, _ in PAIRS:

        a = left["pts"][lm]
        b = right["pts"][lm]

        d = math.hypot(
            a["x"] - b["x"],
            a["y"] - b["y"]
        ) / scale

        distances.append(d)


    # Re-arm when separated
    for i, d in enumerate(distances):

        if d > 0.90:

            pair_state[
                PAIRS[i][1]
            ]["armed"] = True


    best = min(
        range(len(distances)),
        key=lambda i: distances[i]
    )

    minimum = distances[best]


    if minimum > 0.40:
        return None

    if palms < 1.15:
        return None


    # More than one pair closed
    others = [
        d for i, d in enumerate(distances)
        if i != best
    ]

    if min(others) < 0.90:
        return None


    lm, name, output, need_free = PAIRS[best]


    # Y requires thumbs free
    if need_free:

        if (
            left["score"] < RELEASE
            or right["score"] < RELEASE
        ):
            return None


    fire_pair(
        name,
        output[0],
        output[1],
        now
    )

    return name


# ============================================================
# DRAWING
# ============================================================

def point_screen(p, w, h):

    return (
        int((1 - p["x"]) * w),
        int(p["y"] * h)
    )


def draw_hand(frame, hand, now):

    if not hand["pts"]:
        return


    h, w = frame.shape[:2]

    P = [
        point_screen(p, w, h)
        for p in hand["pts"]
    ]


    # --------------------------------------------
    # Skeleton
    # --------------------------------------------

    for a, b in CONNECTIONS:

        cv2.line(
            frame,
            P[a],
            P[b],
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )


    # --------------------------------------------
    # Thumb selector
    # --------------------------------------------

    cv2.line(
        frame,
        P[1],
        P[2],
        (50, 180, 240),
        3,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        P[2],
        P[3],
        (50, 180, 240),
        3,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        P[3],
        P[4],
        (50, 180, 240),
        3,
        cv2.LINE_AA
    )


    stylus = point_screen(
        hand["stylus"],
        w,
        h
    )

    cv2.circle(
        frame,
        stylus,
        7,
        (50, 180, 240),
        -1
    )


    # --------------------------------------------
    # Letters
    # --------------------------------------------

    scale_px = max(
        math.hypot(
            P[0][0] - P[9][0],
            P[0][1] - P[9][1]
        ),
        1
    )


    for i, (lm, finger, segment) in enumerate(TARGETS):

        x, y = P[lm]

        radius = max(
            int(hand["rad"][i] * scale_px),
            12
        )


        near = (
            i == hand["near"]
            and hand["score"] < RELEASE * 1.5
        )


        flash = (
            hand["flash_key"] == i
            and now - hand["flash"] < 0.26
        )


        if flash:
            color = (80, 230, 160)

        elif near:
            color = (50, 180, 240)

        else:
            color = (180, 180, 180)


        cv2.circle(
            frame,
            (x, y),
            4,
            color,
            -1
        )


        letter = LETTERS[
            hand["key"]
        ][i]


        cv2.putText(
            frame,
            letter,
            (x - 10, y - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )


        if near or flash:

            cv2.circle(
                frame,
                (x, y),
                radius,
                color,
                2,
                cv2.LINE_AA
            )


# ============================================================
# UI
# ============================================================

def draw_ui(frame, pair):

    h, w = frame.shape[:2]

    # Title
    cv2.putText(
        frame,
        "THUMBT YPE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2
    )


    # Status panel
    x = w - 300

    cv2.rectangle(
        frame,
        (x, 15),
        (w - 15, 220),
        (10, 25, 40),
        -1
    )


    left = hands["left"]
    right = hands["right"]


    cv2.putText(
        frame,
        "LEFT  : " +
        ("tracking" if left["seen"] else "absent"),
        (x + 15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 150) if left["seen"]
        else (150, 150, 150),
        1
    )


    cv2.putText(
        frame,
        "RIGHT : " +
        ("tracking" if right["seen"] else "absent"),
        (x + 15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 150) if right["seen"]
        else (150, 150, 150),
        1
    )


    active = None

    if left["seen"] and right["seen"]:

        active = (
            left
            if left["score"] < right["score"]
            else right
        )

    elif left["seen"]:
        active = left

    elif right["seen"]:
        active = right


    if active and active["near"] >= 0:

        key = LETTERS[
            active["key"]
        ][active["near"]]

        score = f'{active["score"]:.2f}'

    else:

        key = "-"
        score = "-"


    cv2.putText(
        frame,
        f"KEY   : {key}",
        (x + 15, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1
    )


    cv2.putText(
        frame,
        f"DIST  : {score}",
        (x + 15, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1
    )


    cv2.putText(
        frame,
        f"DEPTH : {'ON' if depth_check else 'OFF'}",
        (x + 15, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1
    )


    cv2.putText(
        frame,
        f"ZONE  : {ZONE_SIZE:.2f}",
        (x + 15, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1
    )


    # Pair command
    if pair:

        cv2.putText(
            frame,
            pair,
            (w // 2 - 50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 230, 160),
            2
        )


    # --------------------------------------------
    # Text box
    # --------------------------------------------

    box_top = h - 145

    cv2.rectangle(
        frame,
        (20, box_top),
        (w - 20, h - 20),
        (5, 15, 25),
        -1
    )


    # Show last lines
    lines = typed_text.split("\n")

    visible = lines[-4:]

    for i, line in enumerate(visible):

        cv2.putText(
            frame,
            line,
            (35, box_top + 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2
        )


    cv2.putText(
        frame,
        "D:Depth  S:Swap  C:Clear  Q:Quit",
        (20, h - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1
    )


# ============================================================
# MEDIAPIPE
# ============================================================

def get_hand_label(result, index):

    try:

        category = result.handedness[index][0]

        label = getattr(
            category,
            "category_name",
            None
        )

        if label:
            return label.lower()

    except Exception:
        pass


    return "right"


def get_points(result, index):

    points = []

    for p in result.hand_landmarks[index]:

        points.append({
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z)
        })

    return points


# ============================================================
# MAIN
# ============================================================

def main():

    global depth_check
    global swap_hands
    global typed_text

    # MediaPipe Tasks
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )


    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )


    detector = vision.HandLandmarker.create_from_options(
        options
    )


    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print("Camera open failed.")
        return


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    print()
    print("=" * 55)
    print("THUMBT YPE STARTED")
    print("=" * 55)
    print("RIGHT HAND : A B C D E F G H I J K L")
    print("LEFT HAND  : M N O P Q R S T U V W X")
    print()
    print("Thumb pad = selector")
    print("Thumb tips = Y")
    print("Index tips = Z")
    print("Middle tips = ENTER")
    print("Little tips = DELETE")
    print("Palms together = SPACE")
    print()
    print("D = Depth")
    print("S = Swap hands")
    print("C = Clear")
    print("Q / ESC = Quit")
    print("=" * 55)


    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break


            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            result = detector.detect(
                mp_image
            )


            now = time.monotonic()


            # Reset hand states
            for side in ["left", "right"]:

                hands[side]["was"] = (
                    hands[side]["seen"]
                )

                hands[side]["seen"] = False

                hands[side]["key"] = side


            # Dark overlay
            output = frame.copy()

            overlay = output.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (
                    output.shape[1],
                    output.shape[0]
                ),
                (8, 23, 38),
                -1
            )

            output = cv2.addWeighted(
                output,
                0.45,
                overlay,
                0.55,
                0
            )


            # ------------------------------------
            # Detect hands
            # ------------------------------------

            if result.hand_landmarks:

                for i in range(
                    len(result.hand_landmarks)
                ):

                    label = get_hand_label(
                        result,
                        i
                    )


                    if swap_hands:

                        if label == "left":
                            label = "right"
                        else:
                            label = "left"


                    if label not in (
                        "left",
                        "right"
                    ):
                        continue


                    if hands[label]["seen"]:
                        continue


                    points = get_points(
                        result,
                        i
                    )


                    measure_hand(
                        hands[label],
                        points
                    )


            # ------------------------------------
            # Pair gestures
            # ------------------------------------

            pair = read_pairs(
                now
            )


            # ------------------------------------
            # Normal typing
            # ------------------------------------

            if hands["left"]["seen"]:

                commit_hand(
                    hands["left"],
                    now,
                    pair is not None
                )


            if hands["right"]["seen"]:

                commit_hand(
                    hands["right"],
                    now,
                    pair is not None
                )


            # ------------------------------------
            # Draw
            # ------------------------------------

            if hands["left"]["seen"]:

                draw_hand(
                    output,
                    hands["left"],
                    now
                )


            if hands["right"]["seen"]:

                draw_hand(
                    output,
                    hands["right"],
                    now
                )


            draw_ui(
                output,
                pair
            )


            cv2.imshow(
                "ThumbType",
                output
            )


            key = cv2.waitKey(1) & 0xFF


            if key in (
                27,
                ord("q"),
                ord("Q")
            ):
                break


            elif key in (
                ord("d"),
                ord("D")
            ):

                depth_check = not depth_check


            elif key in (
                ord("s"),
                ord("S")
            ):

                swap_hands = not swap_hands


            elif key in (
                ord("c"),
                ord("C")
            ):

                typed_text = ""


    finally:

        cap.release()

        cv2.destroyAllWindows()

        detector.close()


        print()
        print("FINAL TEXT:")
        print(typed_text)


if __name__ == "__main__":
    main()