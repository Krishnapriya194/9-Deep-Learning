# -*- coding: utf-8 -*-
"""
AI ROAD SAFETY - COLLISION RISK DETECTION
Fixes:
1. Checks pedestrian risk against CAR, BUS, and TRUCK.
2. Keeps collision alerts visible through brief missed detections.
3. Uses pair-specific risk confirmation and alert hold time.
4. Saves an annotated output video.

NOTE: Pixel distance is not real-world distance. This is a visual risk heuristic,
not a certified collision prediction system.
"""

from ultralytics import YOLO
import cv2
import os
import math
from collections import defaultdict, deque

# ============================================================
# 1. PATHS - CHECK THAT THIS IS THE VIDEO YOU WANT TO ANALYSE
# ============================================================
VIDEO_PATH = r"D:\HOPE AI COURSE\Codings\10) Deep Learning\P1-AI Road Safety\road_video_slow3.mp4"
OUTPUT_PATH = r"D:\HOPE AI COURSE\Codings\10) Deep Learning\P1-AI Road Safety\Videos\road_video_collision_clean_alerts.mp4"
MODEL_PATH = "yolov8n.pt"

# COCO class IDs: person=0, car=2, bus=5, truck=7
PERSON, CAR, BUS, TRUCK = 0, 2, 5, 7
VEHICLES = {CAR: "CAR", BUS: "BUS", TRUCK: "TRUCK"}
CLASSES_TO_TRACK = [PERSON, CAR, BUS, TRUCK]

# ============================================================
# 2. SETTINGS (pixel-based starting values; tune per video)
# ============================================================
CONF = 0.12                 # Slightly lower to recover some missed pedestrians
IMGSZ = 640                  # Keep 640 for reasonable CPU speed
HISTORY_LENGTH = 12

# These are image-space heuristics, not metres. Tune on your own video.
MOVING_PED_MAX_DISTANCE = 1100       # broad detection range for early warning
STATIONARY_PED_MAX_DISTANCE = 850   # information notice range for stationary people
PATH_LOOKAHEAD = 1200               # project farther along vehicle motion
PATH_MARGIN = 95
WARNING_GAP_PX = 950                # show WARNING while pedestrian is still ahead
ALERT_GAP_PX = 500                  # escalate only when projected conflict is nearer
WARNING_TTC_SECONDS = 6.0
ALERT_TTC_SECONDS = 3.5
VEHICLE_MOVEMENT_THRESHOLD = 0.8
PEDESTRIAN_MOVEMENT_THRESHOLD = 1.5

# Trigger a little earlier, while retaining short confirmation and alert hold.
REQUIRED_RISK_FRAMES = 2
ALERT_HOLD_SECONDS = 2.5
RISK_COUNTER_DECAY = 1
EARLY_TTC_SECONDS = 6.0       # Approximate time-to-conflict threshold
MIN_CLOSING_SPEED = 1.0       # Pixels/frame; filters noisy TTC estimates

# ============================================================
# 3. HELPERS
# ============================================================
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def bottom_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int(y2))

def average_step_speed(history):
    if len(history) < 2:
        return 0.0
    steps = [euclidean(history[i - 1], history[i]) for i in range(1, len(history))]
    return sum(steps) / len(steps)

def get_direction(history):
    if len(history) < 3:
        return None
    # Use a longer baseline to reduce jitter.
    x0, y0 = history[0]
    x1, y1 = history[-1]
    dx, dy = x1 - x0, y1 - y0
    magnitude = math.hypot(dx, dy)
    if magnitude < 3:
        return None
    return dx / magnitude, dy / magnitude

def point_to_segment_distance(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return euclidean(point, start)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    closest = (x1 + t * dx, y1 + t * dy)
    return euclidean(point, closest)

def direction_toward(source, target):
    dx, dy = target[0] - source[0], target[1] - source[1]
    mag = math.hypot(dx, dy)
    if mag < 1:
        return None
    return dx / mag, dy / mag

def is_approaching(distance_history):
    # Compare recent distance with earlier distance; allow some frame jitter.
    if len(distance_history) < 4:
        return False
    vals = list(distance_history)
    earlier = sum(vals[:2]) / 2.0
    recent = sum(vals[-2:]) / 2.0
    return recent < earlier - 2.0

def estimate_ttc_seconds(distance_history, fps):
    """
    Rough time-to-contact estimate from image-space distance change.
    This is only a visual heuristic; perspective and camera motion affect it.
    """
    vals = list(distance_history)
    if len(vals) < 4:
        return None
    closing_speed = (vals[0] - vals[-1]) / max(1, len(vals) - 1)
    if closing_speed < MIN_CLOSING_SPEED:
        return None
    return max(0.0, vals[-1] / closing_speed / max(fps, 1.0))

def draw_label(frame, text, x, y, color, scale=0.55, thickness=2):
    y = max(20, y)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

# ============================================================
# 4. INITIALISE
# ============================================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model = YOLO(MODEL_PATH)

print("=" * 62)
print("AI ROAD SAFETY - COLLISION RISK DETECTION")
print("Model classes:", model.names)
print("Tracking person + car + bus + truck")
print("=" * 62)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(
        "ERROR: Could not open the input video. Check VIDEO_PATH:\n" + VIDEO_PATH
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise SystemExit("ERROR: Could not create output video:\n" + OUTPUT_PATH)

# Per-track histories
position_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
distance_history = defaultdict(lambda: deque(maxlen=8))
risk_counter = defaultdict(int)

# Alert memory: pair -> last frame where risk was confirmed, plus details.
active_alerts = {}
hold_frames = max(1, int(ALERT_HOLD_SECONDS * fps))

frame_number = 0
risk_frames = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMGSZ,
            conf=CONF,
            classes=CLASSES_TO_TRACK,
            verbose=False
        )
        result = results[0]
        pedestrians, vehicles = [], []

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                track_id = int(boxes.id[i].item()) if boxes.id is not None else -1
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                box = (x1, y1, x2, y2)
                point = bottom_center(box)

                obj = {
                    "id": track_id, "class": cls_id, "conf": confidence,
                    "box": box, "point": point, "name": "PEDESTRIAN"
                }

                # Do not mix histories of untracked detections.
                if track_id != -1:
                    key_type = "person" if cls_id == PERSON else "vehicle"
                    position_history[(key_type, track_id)].append(point)

                if cls_id == PERSON:
                    pedestrians.append(obj)
                elif cls_id in VEHICLES:
                    obj["name"] = VEHICLES[cls_id]
                    vehicles.append(obj)

        # Movement estimates
        for vehicle in vehicles:
            tid = vehicle["id"]
            hist = position_history[("vehicle", tid)] if tid != -1 else []
            vehicle["speed"] = average_step_speed(hist)
            vehicle["direction"] = get_direction(hist)

        for ped in pedestrians:
            tid = ped["id"]
            hist = position_history[("person", tid)] if tid != -1 else []
            ped["speed"] = average_step_speed(hist)
            ped["moving"] = ped["speed"] >= PEDESTRIAN_MOVEMENT_THRESHOLD
            ped["direction"] = get_direction(hist)

        # Draw all tracked objects.
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["box"]
            name, tid = vehicle["name"], vehicle["id"]
            color = {"CAR": (255, 255, 0), "BUS": (0, 165, 255),
                     "TRUCK": (255, 0, 255)}[name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, f"{name} ID:{tid}", x1, y1 - 8, color)

        for ped in pedestrians:
            x1, y1, x2, y2 = ped["box"]
            color = (0, 255, 255) if ped["moving"] else (210, 210, 210)
            state = "MOVING" if ped["moving"] else "STATIONARY"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, f"PEDESTRIAN ID:{ped['id']} {state}",
                       x1, y1 - 8, color, scale=0.45)

        # Evaluate every pedestrian against every vehicle class.
        frame_alert_candidates = []
        visible_pair_keys = set()

        for ped in pedestrians:
            if ped["id"] == -1:
                continue
            for vehicle in vehicles:
                if vehicle["id"] == -1:
                    continue

                pair_key = (ped["id"], vehicle["id"])
                visible_pair_keys.add(pair_key)
                ped_point = ped["point"]
                veh_point = vehicle["point"]
                distance = euclidean(ped_point, veh_point)
                distance_history[pair_key].append(distance)

                vehicle_moving = vehicle["speed"] >= VEHICLE_MOVEMENT_THRESHOLD
                ped_moving = ped["moving"]

                # Project the pedestrian onto the vehicle's observed direction.
                # A positive forward_gap means the pedestrian is still ahead of
                # the vehicle; after the vehicle has passed, this becomes <= 0,
                # preventing a new alert from first appearing after crossing.
                direction = vehicle["direction"]
                x1, _, x2, _ = vehicle["box"]
                vehicle_half_width = max(35, int((x2 - x1) * 0.40))
                path_width = vehicle_half_width + PATH_MARGIN

                if direction is not None:
                    rel_x = ped_point[0] - veh_point[0]
                    rel_y = ped_point[1] - veh_point[1]
                    forward_gap = rel_x * direction[0] + rel_y * direction[1]
                    lateral_gap = abs(rel_x * (-direction[1]) + rel_y * direction[0])
                    path_end = (
                        int(veh_point[0] + direction[0] * PATH_LOOKAHEAD),
                        int(veh_point[1] + direction[1] * PATH_LOOKAHEAD)
                    )
                    path_distance = point_to_segment_distance(ped_point, veh_point, path_end)
                    in_path = lateral_gap <= path_width and forward_gap > 0
                else:
                    # Before a stable direction is available, use a conservative
                    # proximity fallback, but do not claim a forward projection.
                    forward_gap = distance
                    path_distance = distance
                    in_path = distance <= STATIONARY_PED_MAX_DISTANCE

                approaching = is_approaching(distance_history[pair_key])
                ttc_seconds = estimate_ttc_seconds(distance_history[pair_key], fps)
                vehicle_moving = vehicle["speed"] >= VEHICLE_MOVEMENT_THRESHOLD
                ped_moving = ped["moving"]

                # Stage 1: early warning while the person is ahead and inside
                # the projected corridor. TTC is supplementary, not the only gate.
                projected_warning = (
                    direction is not None and in_path
                    and 0 < forward_gap <= WARNING_GAP_PX
                )
                ttc_warning = (
                    direction is not None and in_path and forward_gap > 0
                    and ttc_seconds is not None
                    and ttc_seconds <= WARNING_TTC_SECONDS
                )
                early_conflict = vehicle_moving and (
                    projected_warning or ttc_warning or
                    (direction is None and approaching and distance <= MOVING_PED_MAX_DISTANCE)
                )

                # Stage 2: stronger alert only when projected conflict is nearer.
                projected_alert = (
                    direction is not None and in_path
                    and 0 < forward_gap <= ALERT_GAP_PX
                )
                ttc_alert = (
                    direction is not None and in_path and forward_gap > 0
                    and ttc_seconds is not None
                    and ttc_seconds <= ALERT_TTC_SECONDS
                )
                moving_risk = ped_moving and early_conflict
                moving_alert = ped_moving and vehicle_moving and (projected_alert or ttc_alert)

                # Only moving pedestrians can trigger a warning or alert.
                # Stationary pedestrians are detected/drawn, but create no panel.
                risk_now = moving_risk
                stage = "ALERT" if moving_alert else "WARNING" if moving_risk else None

                if risk_now:
                    risk_counter[pair_key] += 1
                else:
                    risk_counter[pair_key] = max(
                        0, risk_counter[pair_key] - RISK_COUNTER_DECAY
                    )

                if risk_counter[pair_key] >= REQUIRED_RISK_FRAMES:
                    active_alerts[pair_key] = {
                        "last_frame": frame_number,
                        "distance": distance,
                        "vehicle": vehicle,
                        "pedestrian": ped,
                        "reason": "MOVING PEDESTRIAN IN/NEAR VEHICLE PATH",
                        "ttc": ttc_seconds,
                        "stage": stage or "WARNING",
                        "forward_gap": forward_gap
                    }

        # Remove alerts only after the hold period has elapsed. This prevents
        # one missed detector frame from making the warning flash off.
        expired = [
            key for key, data in active_alerts.items()
            if frame_number - data["last_frame"] > hold_frames
        ]
        for key in expired:
            del active_alerts[key]

        # Keep the display simple: show only the single most urgent active pair.
        # Stationary pedestrians never generate a warning/alert panel.
        visible_alerts = {
            key: data for key, data in active_alerts.items()
            if data.get("pedestrian", {}).get("moving", False)
        }
        if visible_alerts:
            risk_frames += 1
            alert_key, alert = min(
                visible_alerts.items(),
                key=lambda item: (
                    0 if item[1].get("stage") == "ALERT" else 1,
                    item[1].get("ttc") if item[1].get("ttc") is not None else 999.0,
                    item[1].get("forward_gap", 999999)
                )
            )
            vehicle = alert["vehicle"]
            ped = alert["pedestrian"]
            distance = int(alert["distance"])
            alert_stage = alert.get("stage", "WARNING")

            # One compact panel; the same warning panel turns red when escalated.
            panel_color = (0, 0, 255) if alert_stage == "ALERT" else (0, 165, 255)
            title = "COLLISION ALERT - BRAKE / SLOW DOWN" if alert_stage == "ALERT" else "WARNING - PEDESTRIAN AHEAD"
            action_text = "COLLISION RISK: SLOW DOWN / BRAKE" if alert_stage == "ALERT" else "REDUCE SPEED AND BE PREPARED TO BRAKE"

            margin = 20
            panel_w = max(1, min(int(width - 40), 760))
            panel_h = 118
            x, y = 20, 20
            x2 = max(x + 1, min(int(width - 1), int(x + panel_w)))
            y2 = max(y + 1, min(int(height - 1), int(y + panel_h)))
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 0), -1)
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), panel_color, 3)
            draw_label(frame, title, x + 12, y + 27, panel_color, scale=0.58, thickness=2)
            draw_label(frame, f"Vehicle: {vehicle['name']}  |  Pedestrian ID: {ped['id']}",
                       x + 12, y + 53, panel_color, scale=0.48, thickness=1)
            ttc = alert.get("ttc")
            ttc_text = f"  |  Estimated TTC: {ttc:.1f}s" if ttc is not None else ""
            gap = alert.get("forward_gap")
            gap_text = f"  |  Ahead: {int(gap)} px" if gap is not None and gap > 0 else ""
            draw_label(frame, f"Image distance: {distance} px{gap_text}{ttc_text}",
                       x + 12, y + 77, panel_color, scale=0.43, thickness=1)
            draw_label(frame, action_text, x + 12, y + 103, panel_color, scale=0.46, thickness=2)
            cv2.line(frame, ped["point"], vehicle["point"], panel_color, 3)
        else:
            draw_label(frame, "ROAD SAFETY: NO ACTIVE WARNING",
                       30, 55, (0, 220, 0), scale=0.75, thickness=2)

        draw_label(frame, "AI ROAD SAFETY - COLLISION RISK DETECTION",
                   25, height - 25, (255, 255, 255), scale=0.6, thickness=2)

        out.write(frame)

        if frame_number % 100 == 0:
            pct = (frame_number / total_frames * 100) if total_frames else 0
            print(f"Processed {frame_number}/{total_frames} ({pct:.1f}%)")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("\n" + "=" * 62)
print("COLLISION RISK DETECTION FINISHED")
print("Frames processed:", frame_number)
print("Frames with an active alert:", risk_frames)
print("Output video:", OUTPUT_PATH)
print("=" * 62)
