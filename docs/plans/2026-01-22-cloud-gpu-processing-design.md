# Cloud GPU Processing Design

**Date:** 2026-01-22
**Issue:** bbva-gl2
**Status:** Approved

## Problem

SAM3 tracking quality is excellent but requires ~60GB+ memory for long videos. Local processing on a 64GB MacBook causes heavy paging and takes ~25 days for a 1-hour video. Cloud GPU (A100 80GB) would process the same video in ~2-4 hours for ~$2-4.

## Design Goals

1. **Cost efficiency** - Only pay for GPU time when actively processing
2. **Simplicity** - Minimal new infrastructure, easy to operate

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Cloudflare R2                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   videos/   │  │   jobs/     │  │  results/   │  │   status/   │    │
│  │ {job_id}.mp4│  │ {job_id}.json│ │{job_id}.json│  │{job_id}.json│    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
        ▲                  ▲                  │                  │
        │ upload           │ upload           │ download         │ poll
        │                  │                  ▼                  ▼
┌───────┴──────────────────┴───────┐  ┌──────────────────────────────────┐
│         Local Machine            │  │      Cloud GPU (manual start)   │
│  ┌────────────┐  ┌────────────┐  │  │  ┌────────────────────────────┐  │
│  │ FastAPI    │  │ Import CLI │  │  │  │    Cloud Worker            │  │
│  │ (submit)   │  │ (results)  │  │  │  │  - Poll for jobs           │  │
│  └────────────┘  └────────────┘  │  │  │  - Download video          │  │
│         │              │         │  │  │  - Run SAM3                │  │
│         ▼              ▼         │  │  │  - Upload results          │  │
│  ┌────────────────────────────┐  │  │  └────────────────────────────┘  │
│  │      Local SQLite          │  │  └──────────────────────────────────┘
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

**The flow:**
1. Submit job locally → uploads video + job manifest to R2
2. Manually start cloud GPU → worker finds job, processes, uploads results
3. Run import locally → pulls results from R2 into SQLite

## R2 Bucket Structure

```
basketball-analyzer/
├── videos/
│   └── {job_id}.mp4          # Uploaded video file
├── jobs/
│   └── {job_id}.json         # Job manifest (params, status)
├── results/
│   └── {job_id}.json         # Detection results (array of detections)
└── status/
    └── {job_id}.json         # Progress updates (optional, for polling)
```

### Job Manifest (`jobs/{job_id}.json`)

```json
{
  "job_id": "abc-123",
  "video_id": 1,
  "status": "pending",
  "created_at": "2026-01-22T10:00:00Z",
  "parameters": {
    "sample_interval": 1,
    "confidence_threshold": 0.25
  }
}
```

Status transitions: `pending` → `processing` → `completed` → `imported`

### Results File (`results/{job_id}.json`)

```json
{
  "job_id": "abc-123",
  "video_id": 1,
  "completed_at": "2026-01-22T12:30:00Z",
  "frames_processed": 108000,
  "detections": [
    {"frame": 0, "track_id": 1, "bbox": [100, 200, 50, 120], "confidence": 0.92},
    {"frame": 0, "track_id": 2, "bbox": [300, 180, 48, 115], "confidence": 0.88}
  ]
}
```

JSON chosen over SQLite dump for human-readability and no version compatibility issues. ~1M detections ≈ 50-80MB compressed.

## User Workflow

### Step 1: Submit Job (Local)

```bash
# Via API
curl -X POST http://localhost:8000/api/videos/1/detect?target=cloud

# Or CLI
python -m worker.cli submit --video-id 1
```

### Step 2: Start Cloud GPU (Manual)

```bash
# SSH to cloud instance (RunPod/Lambda Labs/Vast.ai)
ssh user@cloud-gpu-instance

# Start worker
docker run -e R2_ACCESS_KEY_ID=xxx -e R2_SECRET_ACCESS_KEY=xxx \
  basketball-analyzer-worker
```

### Step 3: Import Results (Local)

```bash
# Import all completed jobs
python -m worker.cli import --all

# Or import specific job
python -m worker.cli import --job-id abc-123
```

### Step 4: Check Status (Anytime)

```bash
python -m worker.cli status

# Output:
# JOB_ID      VIDEO  STATUS      PROGRESS
# abc-123     1      processing  1200/108000 frames
# def-456     2      completed   ready to import
```

## Implementation Components

### New Files

| File | Purpose |
|------|---------|
| `backend/worker/cloud_storage.py` | R2 upload/download/list operations |
| `backend/worker/cloud_worker.py` | Cloud-specific worker (polls R2 instead of local DB) |
| `backend/worker/cli.py` | CLI commands: `submit`, `import`, `status` |
| `backend/app/services/cloud_job_service.py` | Create cloud jobs, upload video to R2 |
| `Dockerfile.worker` | Container image for cloud deployment |

### Modified Files

| File | Change |
|------|--------|
| `backend/app/api/detection.py` | Add `target=cloud` parameter to submit endpoint |
| `backend/app/config.py` | R2 credentials, bucket name settings |
| `backend/app/models/processing_job.py` | Add `target` field (local/cloud), `r2_video_key` |

### Dependencies

```
boto3          # S3-compatible API for R2
click          # CLI framework
```

## Docker & Model Packaging

SAM3 model is baked into the Docker image to:
1. Avoid downloading on every container start
2. Avoid HuggingFace authentication on cloud worker

### Model Loading Strategy

```python
MODEL_PATHS = [
    "/models/sam3",                    # Baked into container
    "~/.cache/huggingface/hub/...",   # Local cache (dev)
    "facebook/sam3",                   # HuggingFace download (fallback)
]

def load_sam3_model():
    for path in MODEL_PATHS:
        if Path(path).exists():
            return Sam3VideoModel.from_pretrained(path, local_files_only=True)
    return Sam3VideoModel.from_pretrained("facebook/sam3")
```

### Dockerfile.worker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e ".[worker]"

# Copy application code
COPY backend/ .

# Copy pre-downloaded SAM3 model (built locally)
COPY models/sam3 /models/sam3

# Default command
CMD ["python", "-m", "worker", "--cloud"]
```

### Local Build Process

```bash
# 1. Download model locally (one-time, with HF auth)
python -c "from transformers import Sam3VideoModel; Sam3VideoModel.from_pretrained('facebook/sam3')"

# 2. Copy to models/ directory
cp -r ~/.cache/huggingface/hub/models--facebook--sam3 models/sam3

# 3. Build image with model baked in
docker build -f Dockerfile.worker -t basketball-analyzer-worker .

# 4. Push to registry (for cloud pull)
docker push your-registry/basketball-analyzer-worker
```

**Image size estimate:** ~8-10GB (PyTorch + CUDA + SAM3 model)

## Configuration

### New Settings (`backend/app/config.py`)

```python
# Cloud storage (R2)
r2_account_id: str = ""
r2_access_key_id: str = ""
r2_secret_access_key: str = ""
r2_bucket_name: str = "basketball-analyzer"
r2_endpoint_url: str = ""  # https://{account_id}.r2.cloudflarestorage.com

# Cloud worker settings
cloud_worker_poll_interval: float = 10.0  # seconds between R2 polls
cloud_model_path: str = "/models/sam3"    # local path in container
```

### Environment Variables

```bash
# Local .env
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-key
R2_SECRET_ACCESS_KEY=your-secret
R2_BUCKET_NAME=basketball-analyzer

# Cloud worker (passed via docker run -e or cloud provider UI)
WORKER_MODE=cloud
```

No secrets in Docker image - R2 credentials passed at runtime.

## Cost Estimate

- **Storage:** ~40GB/month peak → R2 free tier (10GB) + ~$0.50/month overflow
- **Compute:** ~$2-4 per game (A100 for ~2hrs) → ~$10-20/month for 4-5 games

## Future Improvements

See bbva-qvv for potential migration to Convex.dev cloud database, which would provide:
- Real-time progress updates
- Single source of truth (no file sync)
- Simpler architecture

Blocked until cloud GPU processing is stable and detection volume limits are validated.
