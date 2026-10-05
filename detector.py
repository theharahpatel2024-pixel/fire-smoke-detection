"""
detector.py - Fire and Smoke Detection using classical Image Processing (OpenCV)

Pipeline:
  1. Resize + Gaussian blur            (Exp 2 - spatial filtering)
  2. BGR -> HSV colour conversion
  3. Colour thresholding (cv2.inRange) (fire = red/orange/yellow, smoke = grey)
  4. Morphology: opening + closing     (Exp 4)
  5. Contours + bounding boxes         (Exp 5)
  6. Decision using area percentage
  7. Video only: frame differencing (motion) to confirm smoke
"""
import cv2
import numpy as np

# Default parameters (HSV in OpenCV: H 0-179, S 0-255, V 0-255)
DEFAULTS = dict(
    fire_h_max=35,      # fire hue range: 0..h_max and 170..179 (red wraps around)
    fire_s_min=120,     # fire is strongly coloured (high saturation)
    fire_v_min=150,     # and bright
    smoke_s_max=50,     # smoke is greyish (low saturation)
    smoke_v_min=100,    # not too dark
    smoke_v_max=220,    # not pure white (sky / bright wall)
    min_area_pct=0.5,   # ignore blobs smaller than this % of the image
    blur=5,             # Gaussian kernel size (odd)
    kernel=5,           # morphology kernel size
)

FIRE_COLOR = (0, 0, 255)     # red box  (BGR)
SMOKE_COLOR = (255, 120, 0)  # blue box (BGR)


def resize_keep_ratio(img, max_width=640):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def clean_mask(mask, k):
    """Opening removes small white noise, closing fills small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def filter_contours(mask, min_area_px):
    """Keep only contours bigger than min_area_px. Returns (clean mask, boxes)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area_px:
            cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
            boxes.append(cv2.boundingRect(c))  # (x, y, w, h)
    return out, boxes


def fire_mask_hsv(hsv, p):
    lo1 = np.array([0, p["fire_s_min"], p["fire_v_min"]])
    hi1 = np.array([p["fire_h_max"], 255, 255])
    lo2 = np.array([170, p["fire_s_min"], p["fire_v_min"]])
    hi2 = np.array([179, 255, 255])
    return cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)


def smoke_mask_hsv(hsv, p):
    lo = np.array([0, 0, p["smoke_v_min"]])
    hi = np.array([179, p["smoke_s_max"], p["smoke_v_max"]])
    return cv2.inRange(hsv, lo, hi)


def motion_mask(prev_gray, gray, thresh=15):
    """Frame differencing: white where pixels changed between two frames."""
    diff = cv2.absdiff(prev_gray, gray)
    _, m = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return cv2.dilate(m, np.ones((15, 15), np.uint8))


def detect(img_bgr, params=None, motion=None, detect_fire=True, detect_smoke=True):
    """
    Run detection on one BGR image.
    motion: optional motion mask (video). If given, smoke must also be moving.
    Returns a dict with masks, boxes, percentages, label and annotated image.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    img = resize_keep_ratio(img_bgr)
    h, w = img.shape[:2]
    min_area_px = (p["min_area_pct"] / 100.0) * h * w

    blurred = cv2.GaussianBlur(img, (_odd(p["blur"]), _odd(p["blur"])), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    empty = np.zeros((h, w), np.uint8)
    fire_m, fire_boxes = empty.copy(), []
    smoke_m, smoke_boxes = empty.copy(), []

    if detect_fire:
        raw = clean_mask(fire_mask_hsv(hsv, p), p["kernel"])
        fire_m, fire_boxes = filter_contours(raw, min_area_px)

    if detect_smoke:
        raw = smoke_mask_hsv(hsv, p)
        if motion is not None:
            motion = cv2.resize(motion, (w, h))
            raw = cv2.bitwise_and(raw, motion)
        raw = clean_mask(raw, p["kernel"])
        smoke_m, smoke_boxes = filter_contours(raw, min_area_px)

    fire_pct = 100.0 * np.count_nonzero(fire_m) / (h * w)
    smoke_pct = 100.0 * np.count_nonzero(smoke_m) / (h * w)

    labels = []
    if fire_boxes:
        labels.append("Fire Detected")
    if smoke_boxes:
        labels.append("Smoke Detected")
    label = " + ".join(labels) if labels else "No Fire/Smoke"

    out = img.copy()
    for (x, y, bw, bh) in fire_boxes:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), FIRE_COLOR, 2)
        cv2.putText(out, "Fire", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FIRE_COLOR, 2)
    for (x, y, bw, bh) in smoke_boxes:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), SMOKE_COLOR, 2)
        cv2.putText(out, "Smoke", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SMOKE_COLOR, 2)
    cv2.rectangle(out, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(out, f"{label} | fire {fire_pct:.1f}% smoke {smoke_pct:.1f}%",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return dict(original=img, fire_mask=fire_m, smoke_mask=smoke_m,
                fire_boxes=fire_boxes, smoke_boxes=smoke_boxes,
                fire_pct=fire_pct, smoke_pct=smoke_pct,
                label=label, annotated=out)


def process_video(in_path, out_path, params=None, detect_fire=True, detect_smoke=True,
                  max_frames=300, progress_cb=None):
    """
    Detect frame by frame, write annotated video to out_path.
    Smoke is confirmed with frame differencing (motion).
    Returns a summary dict with sample frames.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    total = min(total, max_frames)

    writer = None
    prev_gray = None
    n = fire_frames = smoke_frames = 0
    samples = []

    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        small = resize_keep_ratio(frame)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mot = motion_mask(prev_gray, gray) if prev_gray is not None else None
        prev_gray = gray

        # first frame has no motion info, so smoke is skipped for that frame only
        res = detect(small, params, motion=mot, detect_fire=detect_fire,
                     detect_smoke=detect_smoke and mot is not None)
        out = res["annotated"]

        if writer is None:
            hh, ww = out.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
        writer.write(out)

        fire_frames += bool(res["fire_boxes"])
        smoke_frames += bool(res["smoke_boxes"])
        if n % max(1, total // 4) == 0 and len(samples) < 4:
            samples.append(out.copy())
        n += 1
        if progress_cb:
            progress_cb(min(n / total, 1.0))

    cap.release()
    if writer is not None:
        writer.release()
    return dict(frames=n, fire_frames=fire_frames, smoke_frames=smoke_frames,
                samples=samples, fps=fps)
