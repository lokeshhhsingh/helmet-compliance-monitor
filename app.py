import os
import uuid
import shutil
import subprocess
import threading

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


def to_h264(src, dst):
    # browsers can't play OpenCV's mp4v output, so re-encode to H.264
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([exe, "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23", dst], capture_output=True)
    return r.returncode == 0


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