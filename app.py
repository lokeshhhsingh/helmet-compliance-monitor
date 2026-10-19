import os
import time
import uuid
import shutil
import subprocess
import threading

import boto3
import cv2
import numpy as np
import imageio_ffmpeg
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import process_video, helmet_model

app = FastAPI()
os.makedirs("jobs", exist_ok=True)

JOBS = {}                       # job id -> status dict
JOB_LOCK = threading.Lock()     # models are shared, so process one video at a time

S3_BUCKET = os.environ.get("S3_BUCKET")
SNS_TOPIC = os.environ.get("SNS_TOPIC_ARN")
DDB_TABLE = os.environ.get("DDB_TABLE", "helmet-violations")
REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")


def to_h264(src, dst):
    # browsers can't play OpenCV's mp4v output, so re-encode to H.264
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([exe, "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23", dst], capture_output=True)
    return r.returncode == 0


def alert_loop(job_id, summary):
    # what the vision result triggers:
    #   confirmed no-helmet -> evidence photo to S3, record in DynamoDB, email through SNS
    #   unsure              -> record in DynamoDB marked needs_review, no email
    if not (S3_BUCKET and SNS_TOPIC):
        return {"enabled": False}
    out = {"enabled": True, "sent": 0, "recorded_for_review": 0, "errors": []}
    try:
        s3 = boto3.client("s3", region_name=REGION)
        table = boto3.resource("dynamodb", region_name=REGION).Table(DDB_TABLE)
        sns = boto3.client("sns", region_name=REGION)
    except Exception as ex:
        out["errors"].append(str(ex))
        print("ALERT LOOP setup error:", ex, flush=True)
        return out

    for e in summary["events"]:
        try:
            name = os.path.basename(e["snapshot"])
            key = f"evidence/{job_id}/{name}"
            s3.upload_file(e["snapshot"], S3_BUCKET, key)
            table.put_item(Item={
                "violation_id": f"{job_id}-r{e['rider_id']}",
                "job_id": job_id,
                "rider_id": e["rider_id"],
                "status": "confirmed",
                "time_s": str(e["time_s"]),
                "second_look": e["second_look"],
                "head_brightness": str(e["head_brightness"]),
                "s3_key": key,
                "created": int(time.time()),
            })
            link = f"{BASE_URL}/files/{job_id}/evidence/{name}"
            sns.publish(
                TopicArn=SNS_TOPIC,
                Subject="Helmet violation confirmed",
                Message=(f"Rider #{e['rider_id']} without a helmet at {e['time_s']} s.\n"
                         f"Second look: {e['second_look']}.\n"
                         f"Evidence photo: {link}\n"
                         f"Stored in S3: s3://{S3_BUCKET}/{key}"))
            out["sent"] += 1
        except Exception as ex:
            out["errors"].append(f"rider {e['rider_id']}: {ex}")
            print("ALERT LOOP error:", ex, flush=True)

    for r in summary.get("review", []):
        try:
            table.put_item(Item={
                "violation_id": f"{job_id}-r{r['rider_id']}",
                "job_id": job_id,
                "rider_id": r["rider_id"],
                "status": "needs_review",
                "reason": r["reason"],
                "head_brightness": str(r["head_brightness"]),
                "created": int(time.time()),
            })
            out["recorded_for_review"] += 1
        except Exception as ex:
            out["errors"].append(f"review rider {r['rider_id']}: {ex}")
            print("ALERT LOOP error:", ex, flush=True)
    return out


def run_job(job_id, input_path):
    job = JOBS[job_id]
    folder = os.path.dirname(input_path)
    raw = os.path.join(folder, "raw.mp4")
    final = os.path.join(folder, "annotated.mp4")
    try:
        with JOB_LOCK:
            job["status"] = "processing"

            def progress(done, total):
                job["progress"] = round(100 * done / max(total, 1))

            summary = process_video(input_path, out_path=raw,
                                    snap_dir=os.path.join(folder, "evidence"),
                                    progress_cb=progress)
        job["status"] = "encoding"
        if to_h264(raw, final):
            os.remove(raw)
        else:
            os.replace(raw, final)

        summary["alerts"] = alert_loop(job_id, summary)     # uses local snapshot paths

        summary["output_video"] = f"/files/{job_id}/annotated.mp4"
        for e in summary["events"]:
            e["snapshot"] = f"/files/{job_id}/evidence/" + os.path.basename(e["snapshot"])
        job["summary"] = summary
        job["progress"] = 100
        job["status"] = "done"
    except Exception as ex:
        job["status"] = "error"
        job["error"] = str(ex)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    folder = os.path.join("jobs", job_id)
    os.makedirs(folder)
    input_path = os.path.join(folder, "input.mp4")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    JOBS[job_id] = {"status": "queued", "progress": 0}
    threading.Thread(target=run_job, args=(job_id, input_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.post("/detect")
def detect(file: UploadFile = File(...)):
    # single-image endpoint, handy for quick API tests
    nparr = np.frombuffer(file.file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    with JOB_LOCK:
        results = helmet_model(frame, verbose=False)
    detections = []
    for box in results[0].boxes:
        detections.append({"class": helmet_model.names[int(box.cls[0])],
                           "confidence": round(float(box.conf[0]), 2),
                           "box": box.xyxy[0].tolist()})
    return {"detections": detections}


app.mount("/files", StaticFiles(directory="jobs"), name="files")
app.mount("/static", StaticFiles(directory="static"), name="static")