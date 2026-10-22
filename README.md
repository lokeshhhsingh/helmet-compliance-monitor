# Helmet Compliance Monitor

Upload traffic footage and the system tracks every person on a two-wheeler, judges each head separately, and flags riders without a helmet. Each confirmed violation produces an evidence photo, a database record and an email alert.

Built for the OpenCV AI Competition 2026.

## What it does

- Detects and tracks motorcycles and people, and treats a person overlapping a motorcycle as a rider.
- Judges **each head separately**, up to 3 per bike (driver plus passengers), labelled `1A`, `1B` and so on. One bike can produce two violations.
- Flags a rider only after a **second, enlarged look agrees**. Unsure cases are marked for human review instead of being accused.
- Saves an annotated video and a result summary, and for each confirmed violation stores a photo in S3, a row in DynamoDB and an email through SNS.

## How the detection works

1. A COCO-pretrained YOLOv8n model (`yolov8n.pt`) detects and tracks people and motorcycles (ByteTrack).
2. A custom model (`best.pt`, classes `helmet` / `no-helmet`) runs on the upper part of each rider's box to find heads.
3. Each head is tracked over time and collects votes. A head must be detected 8 times before it counts as a person, which blocks one-off phantom detections.
4. When at least 70% of a head's detection confidence says `no-helmet` and it has been tracked for about a second, the **second look** runs: head crops spaced about 0.2 s apart are enlarged 2x and re-judged. The violation is confirmed only if at least 3 crops were usable and at least 60% agree.
5. Heads in very dark crops are never confirmed automatically. They go to the review list with the reason "low light".

Result statuses per head: `helmet`, `no-helmet` (confirmed), `uncertain` (leans no-helmet but not confirmed, goes to review) and `checking`.

## Architecture

```
Browser -> FastAPI (uvicorn) on EC2 -> pipeline.py (YOLO + tracking)
                                          |
                          confirmed violation (alert loop)
                    +---------------------+---------------------+
                    v                     v                     v
              S3: evidence photo   DynamoDB: record      SNS: email alert
                                   (helmet-violations)   with evidence link
```

AWS access uses the EC2 instance's IAM role, so no credentials are stored in the code or the repo.

## Example result

Clip with a driver and a bare-headed passenger on one bike (20.4 s, 720p):

| Bikes | People | Helmet | No helmet | Uncertain | Violations |
|-------|--------|--------|-----------|-----------|------------|
| 1 | 2 | 0 | 2 | 0 | 2 (`1A`, `1B`) |

Full output: `results/bare_head_perhead.json`. On the deployed CPU-only server the same clip took 57.5 s from clicking Analyze to Done (about 2.8 times real time).

## Setup

Tested with Python 3.12.3. The pinned versions are in `requirements.txt` (CPU-only PyTorch).

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

`yolov8n.pt` is downloaded automatically by Ultralytics on first run. `best.pt` is included in the repo.

### Run on one video (no AWS needed)

```bash
python pipeline.py path/to/clip.mp4
```

This prints a JSON summary and writes `annotated.mp4` and the evidence photos in `evidence/`.

### Run the web app

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, choose a video and press **Analyze video**.

### Settings for AWS alerts

The alert step reads these environment variables:

| Variable | Meaning | Default |
|----------|---------|---------|
| `S3_BUCKET` | Bucket for evidence photos | none |
| `SNS_TOPIC_ARN` | SNS topic that sends the email | none |
| `DDB_TABLE` | DynamoDB table for violation records | `helmet-violations` |
| `AWS_DEFAULT_REGION` | AWS region | `ap-south-1` |
| `PUBLIC_BASE_URL` | Public address of the app, used to build evidence links in emails | empty |

The IAM role (or credentials) needs permission to write to the bucket, put items in the table and publish to the topic.

## Deployment on EC2

The app runs as a systemd service (`helmet.service`) that starts

```
/home/ubuntu/helmet-compliance-monitor/venv/bin/uvicorn app:app --host 0.0.0.0 --port 80
```

and restarts automatically if it stops. The AWS settings from the table above are kept in a separate override file, `/etc/systemd/system/helmet.service.d/aws.conf`, as `Environment=` lines, so they are not stored in the repository.

To update after pushing changes:

```bash
cd ~/helmet-compliance-monitor && git pull && sudo systemctl restart helmet
```

If the instance is stopped and started again, its public IP changes unless an Elastic IP is attached, so update `PUBLIC_BASE_URL` in `aws.conf` to keep the evidence links in emails working:

```bash
sudo nano /etc/systemd/system/helmet.service.d/aws.conf
sudo systemctl daemon-reload
sudo systemctl restart helmet
```

## Repository layout

```
app.py             FastAPI server and alert loop (S3, DynamoDB, SNS)
pipeline.py        detection, tracking, per-head verdicts, second look
detect.py          early single-image detection script
best.pt            trained helmet / no-helmet model
static/index.html  web page
results/           saved JSON results from test clips
requirements.txt   pinned dependencies
```

## Limitations

- Alerts are sent when a clip finishes processing, not at the instant a violation is detected. The pipeline runs at about 2.8 times real time on a CPU instance, so live camera streams would need a GPU or a faster instance.
- Evidence links go through an open `/files` route over plain HTTP, and the photos show identifiable faces. This is acceptable for a demo with stock footage. A real deployment would need HTTPS, expiring links and face blurring.
- A confirmation can pass on a thin margin (for example 2 of 3 re-checks agreeing), and the exact frame where a violation is confirmed can vary slightly between runs and machines.
- Very dark footage is routed to human review instead of being judged automatically.
- Tested on a small set of clips (bare-headed pair, night rider, two helmeted riders). It has not been evaluated on a large labelled traffic dataset.
- At most 3 heads are tracked per bike, and a head partly hidden behind another rider may be missed in frames where it is not visible.
