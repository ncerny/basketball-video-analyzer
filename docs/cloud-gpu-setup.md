# Cloud GPU Worker Setup

This guide covers setting up cloud GPU workers for SAM2 video detection processing.

## Prerequisites

- R2 bucket configured (see [R2 Setup](#r2-setup))
- Docker installed locally
- Access to a container registry (Docker Hub, GHCR, etc.)
- Cloud GPU provider account (RunPod, Lambda Labs, Vast.ai, etc.)

## R2 Setup

1. **Create R2 Bucket** in Cloudflare dashboard:
   - Go to **R2 Object Storage** → **Create bucket**
   - Name it `basketball-analyzer` (or customize)

2. **Create API Token**:
   - Go to **R2** → **Manage R2 API Tokens** → **Create API token**
   - Select **Object Read & Write** permission
   - Copy the credentials

3. **Get Account ID** from dashboard URL:
   ```
   https://dash.cloudflare.com/<ACCOUNT_ID>/r2/overview
   ```

4. **Add to `backend/.env`**:
   ```bash
   R2_ACCOUNT_ID=your-32-char-account-id
   R2_ACCESS_KEY_ID=your-access-key-id
   R2_SECRET_ACCESS_KEY=your-secret-access-key
   R2_BUCKET_NAME=basketball-analyzer
   ```

5. **Test connection**:
   ```bash
   cd backend
   poetry run python -m worker.cli status
   # Should show "No cloud jobs found."
   ```

## Building the Docker Image

### Step 1: Download SAM2 Model

The model needs to be baked into the Docker image to avoid HuggingFace authentication on the cloud worker.

```bash
python -c "from transformers import Sam2VideoModel; Sam2VideoModel.from_pretrained('facebook/sam2-hiera-large')"
```

This downloads ~6GB to your HuggingFace cache.

### Step 2: Copy Model for Docker Build

```bash
cd backend
mkdir -p models/sam2

# Copy from HuggingFace cache
cp -r ~/.cache/huggingface/hub/models--facebook--sam2-hiera-large/snapshots/* models/sam2/
```

### Step 3: Build Docker Image

```bash
cd backend
docker build -f Dockerfile.worker -t basketball-analyzer-worker .
```

**Note:** The image will be ~10-15GB due to PyTorch + CUDA + SAM2 model.

### Step 4: Push to Container Registry

**Docker Hub:**
```bash
docker tag basketball-analyzer-worker your-dockerhub-user/basketball-analyzer-worker
docker push your-dockerhub-user/basketball-analyzer-worker
```

**GitHub Container Registry:**
```bash
docker tag basketball-analyzer-worker ghcr.io/ncerny/basketball-analyzer-worker
docker push ghcr.io/ncerny/basketball-analyzer-worker
```

## Running on Cloud GPU

### Option A: RunPod

1. Go to [runpod.io](https://runpod.io) and create a pod
2. Select GPU: **A100 80GB** (~$1.19/hr) recommended for long videos
3. Use your Docker image as the container
4. Set environment variables in pod config:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET_NAME`
5. The worker starts automatically and polls R2 for jobs

### Option B: Lambda Labs / Vast.ai / SSH

```bash
ssh user@cloud-gpu-instance

docker run --gpus all \
  -e R2_ACCOUNT_ID=xxx \
  -e R2_ACCESS_KEY_ID=xxx \
  -e R2_SECRET_ACCESS_KEY=xxx \
  -e R2_BUCKET_NAME=basketball-analyzer \
  your-registry/basketball-analyzer-worker
```

### Single Job Mode

To process one job and exit (useful for spot instances):

```bash
docker run --gpus all \
  -e R2_ACCOUNT_ID=xxx \
  -e R2_ACCESS_KEY_ID=xxx \
  -e R2_SECRET_ACCESS_KEY=xxx \
  -e R2_BUCKET_NAME=basketball-analyzer \
  -e SINGLE_JOB=true \
  your-registry/basketball-analyzer-worker
```

## End-to-End Workflow

### 1. Submit Job (Local)

```bash
cd backend

# Submit a video for cloud processing
poetry run python -m worker.cli submit \
  --video-id 1 \
  --video-path /path/to/game.mp4

# Output: Job submitted: abc-123-def-456
```

### 2. Check Status

```bash
poetry run python -m worker.cli status

# Output:
# JOB_ID                                 VIDEO  STATUS       PROGRESS
# --------------------------------------------------------------------------------
# abc-123-def-456                        1      pending
```

### 3. Start Cloud Worker

Start your cloud GPU instance. The worker will automatically:
- Poll R2 for pending jobs
- Download the video
- Run SAM2 detection
- Upload results
- Mark job as completed

### 4. Monitor Progress

```bash
poetry run python -m worker.cli status

# Output:
# JOB_ID                                 VIDEO  STATUS       PROGRESS
# --------------------------------------------------------------------------------
# abc-123-def-456                        1      processing   1200/108000 frames
```

### 5. Import Results

Once the job shows `completed`:

```bash
# Import all completed jobs
poetry run python -m worker.cli import-all

# Or import a specific job
poetry run python -m worker.cli import-job --job-id abc-123-def-456
```

This downloads results from R2 and inserts detections into your local SQLite database.

## CLI Reference

| Command | Description |
|---------|-------------|
| `python -m worker.cli submit` | Submit video for cloud processing |
| `python -m worker.cli status` | Show status of all cloud jobs |
| `python -m worker.cli import-job --job-id ID` | Import specific job results |
| `python -m worker.cli import-all` | Import all completed jobs |

### Submit Options

```bash
python -m worker.cli submit \
  --video-id 1 \              # Database video ID (required)
  --video-path /path/to.mp4 \ # Path to video file (required)
  --sample-interval 1 \       # Process every Nth frame (default: 1)
  --confidence 0.25           # Confidence threshold (default: 0.25)
```

## Cost Estimates

| Provider | GPU | $/hour | 1-hour video |
|----------|-----|--------|--------------|
| RunPod | A100 80GB | ~$1.19 | ~$2-4 |
| Lambda Labs | A100 80GB | ~$1.10 | ~$2-4 |
| Vast.ai | A100 80GB | Variable | ~$1-3 |

Processing time for a 1-hour video: ~2-4 hours on A100.

## Troubleshooting

### "No module named 'boto3'"

Use `poetry run` prefix:
```bash
poetry run python -m worker.cli status
```

### Worker can't connect to R2

Check environment variables are set correctly:
```bash
docker run ... -e R2_ACCOUNT_ID=xxx ...
```

### Job stuck in "processing"

Check cloud worker logs. Common issues:
- Out of GPU memory (try A100 80GB)
- Network timeout downloading video
- R2 credentials expired

### Import fails with "Video not found"

The video must exist in your local database before importing. The `--video-id` used during submit must match an existing video record.
