import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

helmet_model = YOLO("best.pt")      # your trained model: helmet / no-helmet
scene_model = YOLO("yolov8n.pt")    # COCO model: people + motorcycles
PERSON, MOTORCYCLE = 0, 3

MIN_NO_HELMET = 8      # no-helmet detections needed before a head becomes a candidate
MIN_SHARE = 0.7        # share of detection confidence that must say "no-helmet"
LOW_LIGHT = 45         # head crops darker than this (0-255) are too dark to judge; 0 turns the rule off
MAX_HEADS = 3          # heads tracked per bike (driver plus passengers)
HEAD_MIN = 8           # a head must be detected this many times to count as a person

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


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def on_a_person(c, boxes):
    # a head only counts if its centre sits in the upper part of a person detected on a bike;
    # this rejects foliage, signs and posters that the helmet model mistakes for heads
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if (b[0] - 0.05 * bw <= c[0] <= b[2] + 0.05 * bw) and (b[1] <= c[1] <= b[1] + 0.6 * bh):
            return True
    return False


def head_region(box, W, H):
    # upper part of the rider's box, where the heads should be
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (max(0, int(x1 - 0.1 * w)), max(0, int(y1 - 0.05 * h)),
            min(W, int(x2 + 0.1 * w)), min(H, int(y2 * 0 + y1 + 0.45 * h)))


def find_heads(frame, region, conf, imgsz=416):
    # run the helmet model on the head region and return every head it finds
    rx1, ry1, rx2, ry2 = region
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return []
    crop = frame[ry1:ry2, rx1:rx2]
    view = brighten(crop) if brightness(crop) < 90 else crop
    r = helmet_model(view, conf=conf, imgsz=imgsz, verbose=False)[0]
    H, W = frame.shape[:2]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        box = (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
        w, h = box[2] - box[0], box[3] - box[1]
        px, py = int(0.6 * w), int(0.6 * h)          # keep some context around the head
        cx1, cy1 = max(0, int(box[0]) - px), max(0, int(box[1]) - py)
        cx2, cy2 = min(W, int(box[2]) + px), min(H, int(box[3]) + py)
        hc = frame[cy1:cy2, cx1:cx2]
        if hc.size == 0:
            continue
        out.append({"cls": helmet_model.names[int(b.cls[0])], "conf": float(b.conf[0]),
                    "box": box, "crop": hc.copy(), "lum": brightness(hc)})
    out.sort(key=lambda d: -d["conf"])               # most confident first
    return out


def second_look(items, conf=0.4):
    # re-check stored head crops at 2x size; confirm only if most still say no-helmet
    agree, seen = 0, 0
    for crop, lum in items:
        view = brighten(crop) if lum < 90 else crop
        big = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r = helmet_model(big, conf=conf, imgsz=640, verbose=False)[0]
        if len(r.boxes) == 0:
            continue
        ch, cw = big.shape[:2]

        def dist(b):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            return ((x1 + x2) / 2 - cw / 2) ** 2 + ((y1 + y2) / 2 - ch / 2) ** 2

        b = min(r.boxes, key=dist)                   # judge the head nearest the crop centre
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


class HeadState:
    def __init__(self, bike_id, slot, frame_idx, box):
        self.label = f"{bike_id}{chr(65 + slot)}"
        self.first = frame_idx
        self.last = frame_idx
        self.box = box
        self.width = box[2] - box[0]
        self.score = {"helmet": 0.0, "no-helmet": 0.0}
        self.count = {"helmet": 0, "no-helmet": 0}
        self.crops = deque(maxlen=10)       # (crop, brightness) of no-helmet detections
        self.last_crop = -999
        self.next_check = MIN_NO_HELMET
        self.confirmed = False
        self.dark = False
        self.reason = ""

    def detections(self):
        return self.count["helmet"] + self.count["no-helmet"]

    def share_no(self):
        total = self.score["helmet"] + self.score["no-helmet"]
        return self.score["no-helmet"] / total if total else 0.0

    def median_light(self):
        return float(np.median([l for _, l in self.crops])) if self.crops else 255.0

    def status(self):
        if self.detections() == 0:
            return "checking"
        if self.confirmed:
            return "no-helmet"
        if self.dark or self.share_no() >= 0.5:
            return "uncertain"       # leaning no-helmet but not confirmed: goes to review
        return "helmet"


class RiderState:
    def __init__(self, bike_id, frame_idx, box):
        self.id = bike_id
        self.first = frame_idx
        self.last = frame_idx
        self.box = box
        self.heads = []


def assign_heads(st, dets, frame_idx, fps):
    # match this frame's heads to the bike's head slots by the best overall fit
    pairs = []
    for di, d in enumerate(dets):
        c = center(d["box"])
        w = d["box"][2] - d["box"][0]
        for si, h in enumerate(st.heads):
            hc = center(h.box)
            dist = ((c[0] - hc[0]) ** 2 + (c[1] - hc[1]) ** 2) ** 0.5
            ref = max(w, h.width)
            limit = 1.5 * ref * (1 + (frame_idx - h.last) / fps)   # slots not seen lately may have moved
            if dist <= limit:
                pairs.append((dist / ref, di, si))
    pairs.sort()
    used_d, used_s, out = set(), set(), []
    for _, di, si in pairs:
        if di in used_d or si in used_s:
            continue
        used_d.add(di)
        used_s.add(si)
        out.append((di, si))

    for di, d in enumerate(dets):                    # heads that fit no slot (most confident first)
        if di in used_d:
            continue
        c = center(d["box"])
        free = [i for i in range(len(st.heads)) if i not in used_s]
        if free:
            # a slot not seen this frame: assume the same person jumped, so take the nearest one
            si = min(free, key=lambda i: (center(st.heads[i].box)[0] - c[0]) ** 2
                     + (center(st.heads[i].box)[1] - c[1]) ** 2)
        elif len(st.heads) < MAX_HEADS:
            # every slot is already matched in this frame, so this must be another person
            st.heads.append(HeadState(st.id, len(st.heads), frame_idx, d["box"]))
            si = len(st.heads) - 1
        else:
            continue
        used_d.add(di)
        used_s.add(si)
        out.append((di, si))

    res = []
    for di, si in out:
        h, d = st.heads[si], dets[di]
        h.box, h.width, h.last = d["box"], d["box"][2] - d["box"][0], frame_idx
        res.append((d, h))
    return res


def match_previous(box, frame_idx, riders, used_now, fps):
    # a new tracker id close to where a bike just vanished is the same bike
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
    min_seen = int(fps)                             # a head must be tracked ~1 second to be judged
    crop_gap = max(2, int(fps * 0.2))               # keep crops about 0.2 s apart
    recent = max(3, int(fps * 0.6))                 # draw a head only if seen this recently
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    riders = {}      # clean bike id -> RiderState
    canon = {}       # raw tracker id -> clean bike id
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
        candidates.sort(key=lambda t: area(t[1]), reverse=True)
        person_boxes = [b for _, b in candidates]   # every person on a bike, nested ones included
        # drop duplicate boxes that sit almost entirely inside a larger rider box
        kept = []
        for raw, box in candidates:
            if any(inter(box, k) / max(area(box), 1) > 0.8 for _, k in kept):
                continue
            kept.append((raw, box))

        # bikes the tracker already knows go first, then new tracker ids
        kept.sort(key=lambda t: t[0] not in canon)
        used_now = set()
        claimed = []     # head centres already taken in this frame

        for raw, box in kept:
            cid = canon.get(raw)
            if cid is None:
                cid = match_previous(box, frame_idx, riders, used_now, fps)
                if cid is None:
                    cid = len(riders) + 1
                    riders[cid] = RiderState(cid, frame_idx, box)
                canon[raw] = cid
            if cid in used_now:
                continue
            used_now.add(cid)
            st = riders[cid]
            st.last, st.box = frame_idx, box

            dets = []
            for d in find_heads(frame, head_region(box, W, H), conf):
                c = center(d["box"])
                w = d["box"][2] - d["box"][0]
                if not on_a_person(c, person_boxes):
                    continue                         # not on a rider: foliage, signs, posters
                if any(((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2) ** 0.5 < 0.6 * max(w, k[2])
                       for k in claimed):
                    continue                         # duplicate box, or a head that belongs to another bike
                claimed.append((c[0], c[1], w))
                dets.append(d)

            for d, h in assign_heads(st, dets, frame_idx, fps):
                h.score[d["cls"]] += d["conf"]
                h.count[d["cls"]] += 1
                if d["cls"] == "no-helmet" and d["conf"] >= 0.4 and frame_idx - h.last_crop >= crop_gap:
                    h.crops.append((d["crop"], d["lum"]))
                    h.last_crop = frame_idx

            # second look, per head: only when the evidence leans strongly towards no-helmet
            newly = []
            for h in st.heads:
                if h.last != frame_idx:
                    continue
                if (not h.confirmed and not h.dark and h.count["no-helmet"] >= h.next_check
                        and h.share_no() >= MIN_SHARE and frame_idx - h.first + 1 >= min_seen):
                    if h.median_light() < LOW_LIGHT:
                        h.dark = True                # too dark to accuse: route to review
                        h.reason = "low light"
                    else:
                        confirmed, agree, seen = second_look(list(h.crops))
                        if confirmed:
                            h.confirmed = True
                            newly.append((h, agree, seen))
                        elif seen < 3:
                            h.reason = "not enough clear views"
                            h.next_check += 4
                        else:
                            h.reason = "re-check disagreed"
                            h.next_check += 10

            # draw the bike and each head that counts as a person and was seen recently
            counted = [h for h in st.heads
                       if h.detections() >= HEAD_MIN and frame_idx - h.last <= recent]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (170, 170, 170), 1)
            label = f"Bike #{cid}" + (f" ({len(counted)} people)" if len(counted) > 1 else "")
            cv2.putText(frame, label, (int(box[0]), int(box[3]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            for h in counted:
                status = h.status()
                color = colors[status]
                hb = h.box
                cv2.rectangle(frame, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])), color, 2)
                ty = int(hb[1]) - 6 if hb[1] > 22 else int(hb[3]) + 16
                cv2.putText(frame, f"{h.label} {status}", (int(hb[0]), ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for hh, agree, seen in newly:
                snap = os.path.join(snap_dir, f"violation_{hh.label}_f{frame_idx}.jpg")
                cv2.imwrite(snap, frame)
                event = {"rider_id": hh.label, "bike_id": cid, "frame": frame_idx,
                         "time_s": round(frame_idx / fps, 2), "snapshot": snap,
                         "second_look": f"{agree}/{seen} re-checks agreed",
                         "head_brightness": round(hh.median_light(), 1)}
                events.append(event)
                if violation_cb:
                    violation_cb(event)

        writer.write(frame)
        frame_idx += 1
        if progress_cb:
            progress_cb(frame_idx, total)

    cap.release()
    writer.release()

    counted_bikes = [st for st in riders.values() if st.last - st.first + 1 >= min_seen]
    persons = [h for st in counted_bikes for h in st.heads if h.detections() >= HEAD_MIN]
    for h in persons:
        save_strip(os.path.join(snap_dir, f"crops_{h.label}.jpg"), list(h.crops))
    statuses = [h.status() for h in persons]
    review = [{"rider_id": h.label, "reason": h.reason or "mixed evidence",
               "head_brightness": round(h.median_light(), 1)}
              for h in persons if h.status() == "uncertain"]
    return {"frames": frame_idx, "duration_s": round(frame_idx / fps, 1),
            "bikes_tracked": len(counted_bikes),
            "riders_tracked": len(persons),          # people on two-wheelers
            "helmet": statuses.count("helmet"),
            "no_helmet": statuses.count("no-helmet"),
            "uncertain": statuses.count("uncertain") + statuses.count("checking"),
            "review": review,
            "violations": len(events), "events": events, "output_video": out_path}


if __name__ == "__main__":
    import sys, json
    print(json.dumps(process_video(sys.argv[1]), indent=2))