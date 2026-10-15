import os
import cv2
from collections import defaultdict
from ultralytics import YOLO

helmet_model = YOLO("best.pt")      # your trained model: helmet / no-helmet
scene_model = YOLO("yolov8n.pt")    # COCO model: people + motorcycles
PERSON, MOTORCYCLE = 0, 3


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
    # crop the head area and run the helmet model on the enlarged crop
    rx1, ry1, rx2, ry2 = region
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return None
    r = helmet_model(frame[ry1:ry2, rx1:rx2], conf=conf, imgsz=imgsz, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    b = max(r.boxes, key=lambda x: float(x.conf[0]))
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    return (helmet_model.names[int(b.cls[0])], float(b.conf[0]),
            (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1))


def process_video(src, out_path="annotated.mp4", conf=0.25, min_frames=8,
                  snap_dir="evidence", progress_cb=None):
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

    votes = defaultdict(lambda: defaultdict(int))   # rider id -> class -> frames
    logged, events = set(), []
    spans = {}                                      # rider id -> [first frame, last frame]
    frame_idx = 0

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
        candidates = [(rid, box) for rid, box in people
                      if any(inter(box, m) / max(area(box), 1) > 0.25 for m in bikes)]

        # drop duplicate boxes that sit almost entirely inside a larger rider box
        candidates.sort(key=lambda t: area(t[1]), reverse=True)
        riders = []
        for rid, box in candidates:
            if any(inter(box, kept) / max(area(box), 1) > 0.8 for _, kept in riders):
                continue
            riders.append((rid, box))

        for rid, box in riders:
            spans.setdefault(rid, [frame_idx, frame_idx])[1] = frame_idx

            found = check_helmet(frame, head_region(box, W, H), conf)
            if found:
                cls, c, hb = found
                votes[rid][cls] += 1
                cv2.rectangle(frame, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])),
                              (255, 255, 255), 1)

            v = votes[rid]
            if v["helmet"] + v["no-helmet"] == 0:
                label, color = "checking", (180, 180, 180)
            elif v["no-helmet"] > v["helmet"]:
                label, color = "no-helmet", (0, 0, 255)
            else:
                label, color = "helmet", (0, 200, 0)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"Rider #{rid}: {label}", (int(box[0]), int(box[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            tracked_long_enough = frame_idx - spans[rid][0] + 1 >= min_seen
            if (label == "no-helmet" and v["no-helmet"] >= min_frames
                    and tracked_long_enough and rid not in logged):
                logged.add(rid)
                snap = os.path.join(snap_dir, f"violation_rider{rid}_f{frame_idx}.jpg")
                cv2.imwrite(snap, frame)
                events.append({"rider_id": rid, "frame": frame_idx,
                               "time_s": round(frame_idx / fps, 2), "snapshot": snap})

        writer.write(frame)
        frame_idx += 1
        if progress_cb:
            progress_cb(frame_idx, total)

    cap.release()
    writer.release()

    def final(rid):
        v = votes[rid]
        if v["helmet"] + v["no-helmet"] == 0:
            return "unknown"
        return "no-helmet" if v["no-helmet"] > v["helmet"] else "helmet"

    real = [r for r, (a, b) in spans.items() if b - a + 1 >= min_seen]
    finals = [final(r) for r in real]
    return {"frames": frame_idx, "duration_s": round(frame_idx / fps, 1),
            "riders_tracked": len(real),
            "helmet": finals.count("helmet"), "no_helmet": finals.count("no-helmet"),
            "unknown": finals.count("unknown"),
            "rider_spans": spans,
            "violations": len(events), "events": events, "output_video": out_path}


if __name__ == "__main__":
    import sys, json
    print(json.dumps(process_video(sys.argv[1]), indent=2))