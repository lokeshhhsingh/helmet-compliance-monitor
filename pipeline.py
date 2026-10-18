import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

helmet_model = YOLO("best.pt")      # your trained model: helmet / no-helmet
scene_model = YOLO("yolov8n.pt")    # COCO model: people + motorcycles
PERSON, MOTORCYCLE = 0, 3

MIN_NO_HELMET = 8      # no-helmet detections needed before a rider becomes a candidate
MIN_SHARE = 0.7        # share of detection confidence that must say "no-helmet"
LOW_LIGHT = 45         # head crops darker than this (0-255) are too dark to judge; 0 turns the rule off

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


def brighten(crop):
    # lift local contrast so the model can see the head shape in dark crops
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def brightness(crop):
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def head_region(box, W, H):
    # upper part of the rider's box, where the head should be
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (max(0, int(x1 - 0.1 * w)), max(0, int(y1 - 0.05 * h)),
            min(W, int(x2 + 0.1 * w)), min(H, int(y1 + 0.45 * h)))


def check_helmet(frame, region, conf, imgsz=320):
    # crop the head area, brighten it if dark, and run the helmet model on it
    rx1, ry1, rx2, ry2 = region
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return None
    crop = frame[ry1:ry2, rx1:rx2]
    lum = brightness(crop)
    view = brighten(crop) if lum < 90 else crop
    r = helmet_model(view, conf=conf, imgsz=imgsz, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    b = max(r.boxes, key=lambda x: float(x.conf[0]))
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    return (helmet_model.names[int(b.cls[0])], float(b.conf[0]),
            (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1), crop.copy(), lum)


def second_look(items, conf=0.4):
    # re-check stored head crops at 2x size; confirm only if most still say no-helmet
    agree, seen = 0, 0
    for crop, lum in items:
        view = brighten(crop) if lum < 90 else crop
        big = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r = helmet_model(big, conf=conf, imgsz=640, verbose=False)[0]
        if len(r.boxes) == 0:
            continue
        b = max(r.boxes, key=lambda x: float(x.conf[0]))
        seen += 1
        if helmet_model.names[int(b.cls[0])] == "no-helmet":
            agree += 1
    return (seen >= 3 and agree / seen >= 0.6), agree, seen


def save_strip(path, items):
    # save the head crops the model judged, side by side, for human inspection
    crops = [c for c, _ in items][-6:]
    if not crops:
        return
    tiles = [cv2.resize(c, (max(1, int(c.shape[1] * 120 / c.shape[0])), 120)) for c in crops]
    cv2.imwrite(path, cv2.hconcat(tiles))


class RiderState:
    def __init__(self, frame_idx, box):
        self.first = frame_idx
        self.last = frame_idx
        self.box = box
        self.score = {"helmet": 0.0, "no-helmet": 0.0}
        self.count = {"helmet": 0, "no-helmet": 0}
        self.crops = deque(maxlen=10)       # (crop, brightness) of no-helmet detections
        self.next_check = MIN_NO_HELMET
        self.confirmed = False
        self.dark = False
        self.reason = ""

    def share_no(self):
        total = self.score["helmet"] + self.score["no-helmet"]
        return self.score["no-helmet"] / total if total else 0.0

    def median_light(self):
        return float(np.median([l for _, l in self.crops])) if self.crops else 255.0

    def status(self):
        if self.count["helmet"] + self.count["no-helmet"] == 0:
            return "checking"
        if self.confirmed:
            return "no-helmet"
        if self.dark or self.share_no() >= 0.5:
            return "uncertain"       # leaning no-helmet but not confirmed: goes to review
        return "helmet"


def match_previous(box, frame_idx, riders, used_now, fps):
    # a new tracker id close to where a rider just vanished is the same rider
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w = box[2] - box[0]
    best, best_d = None, None
    for cid, st in riders.items():
        gap = frame_idx - st.last
        if cid in used_now or gap <= 0 or gap > 1.5 * fps:
            continue
        ox, oy = (st.box[0] + st.box[2]) / 2, (st.box[1] + st.box[3]) / 2
        ow = st.box[2] - st.box[0]
        d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        if d <= max(w, ow) * (1.0 + 1.5 * gap / fps) and (best_d is None or d < best_d):
            best, best_d = cid, d
    return best


def process_video(src, out_path="annotated.mp4", conf=0.25, snap_dir="evidence",
                  progress_cb=None, violation_cb=None):
    os.makedirs(snap_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960
    sh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540
    scale = 960 / max(sw, sh)                       # longest side becomes 960 px
    W, H = int(sw * scale) // 2 * 2, int(sh * scale) // 2 * 2
    min_seen = int(fps)                             # a rider must be tracked ~1 second to count
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    riders = {}      # clean rider id -> RiderState
    canon = {}       # raw tracker id -> clean rider id
    events = []
    frame_idx = 0
    colors = {"helmet": (0, 200, 0), "no-helmet": (0, 0, 255),
              "uncertain": (0, 165, 255), "checking": (180, 180, 180)}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))

        res = scene_model.track(frame, persist=(frame_idx > 0), conf=0.3,
                                classes=[PERSON, MOTORCYCLE],
                                tracker="bytetrack.yaml", verbose=False)[0]
        people, bikes = [], []
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            if int(b.cls[0]) == MOTORCYCLE:
                bikes.append(xyxy)
            elif b.id is not None:
                people.append((int(b.id[0]), xyxy))

        # a rider = a tracked person overlapping a motorcycle
        candidates = [(raw, box) for raw, box in people
                      if any(inter(box, m) / max(area(box), 1) > 0.25 for m in bikes)]
        # drop duplicate boxes that sit almost entirely inside a larger rider box
        candidates.sort(key=lambda t: area(t[1]), reverse=True)
        kept = []
        for raw, box in candidates:
            if any(inter(box, k) / max(area(box), 1) > 0.8 for _, k in kept):
                continue
            kept.append((raw, box))

        # riders the tracker already knows go first, then new tracker ids
        kept.sort(key=lambda t: t[0] not in canon)
        used_now = set()

        for raw, box in kept:
            cid = canon.get(raw)
            if cid is None:
                cid = match_previous(box, frame_idx, riders, used_now, fps)
                if cid is None:
                    cid = len(riders) + 1
                    riders[cid] = RiderState(frame_idx, box)
                canon[raw] = cid
            if cid in used_now:
                continue
            used_now.add(cid)
            st = riders[cid]
            st.last, st.box = frame_idx, box

            found = check_helmet(frame, head_region(box, W, H), conf)
            if found:
                cls, c, hb, crop, lum = found
                st.score[cls] += c
                st.count[cls] += 1
                if cls == "no-helmet" and c >= 0.4:
                    st.crops.append((crop, lum))
                cv2.rectangle(frame, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])),
                              (255, 255, 255), 1)

            # second look: only when the evidence leans strongly towards no-helmet
            newly = False
            agree = seen = 0
            if (not st.confirmed and not st.dark and st.count["no-helmet"] >= st.next_check
                    and st.share_no() >= MIN_SHARE
                    and frame_idx - st.first + 1 >= min_seen):
                if st.median_light() < LOW_LIGHT:
                    st.dark = True                       # too dark to accuse: route to review
                    st.reason = "low light"
                else:
                    confirmed, agree, seen = second_look(list(st.crops))
                    if confirmed:
                        st.confirmed = True
                        newly = True
                    else:
                        st.reason = "re-check disagreed"
                        st.next_check += 10

            status = st.status()
            color = colors[status]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"Rider #{cid}: {status}", (int(box[0]), int(box[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if newly:
                snap = os.path.join(snap_dir, f"violation_rider{cid}_f{frame_idx}.jpg")
                cv2.imwrite(snap, frame)
                event = {"rider_id": cid, "frame": frame_idx,
                         "time_s": round(frame_idx / fps, 2), "snapshot": snap,
                         "second_look": f"{agree}/{seen} re-checks agreed",
                         "head_brightness": round(st.median_light(), 1)}
                events.append(event)
                if violation_cb:
                    violation_cb(event)

        writer.write(frame)
        frame_idx += 1
        if progress_cb:
            progress_cb(frame_idx, total)

    cap.release()
    writer.release()

    real = [cid for cid, st in riders.items() if st.last - st.first + 1 >= min_seen]
    for cid in real:
        save_strip(os.path.join(snap_dir, f"crops_rider{cid}.jpg"), list(riders[cid].crops))
    statuses = [riders[c].status() for c in real]
    review = [{"rider_id": c, "reason": riders[c].reason or "mixed evidence",
               "head_brightness": round(riders[c].median_light(), 1)}
              for c in real if riders[c].status() == "uncertain"]
    return {"frames": frame_idx, "duration_s": round(frame_idx / fps, 1),
            "riders_tracked": len(real),
            "helmet": statuses.count("helmet"),
            "no_helmet": statuses.count("no-helmet"),
            "uncertain": statuses.count("uncertain") + statuses.count("checking"),
            "review": review,
            "violations": len(events), "events": events, "output_video": out_path}


if __name__ == "__main__":
    import sys, json
    print(json.dumps(process_video(sys.argv[1]), indent=2))