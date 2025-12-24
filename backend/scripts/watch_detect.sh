#!/bin/bash

status="processing"

seconds=${1:-60}

job_id=$(curl -X POST "http://localhost:8000/api/videos/7/detect" \
  -H "Content-Type: application/json" \
  -d "{\"sample_interval\": 1, \"max_seconds\": ${seconds}}"  2>/dev/null | jq -r .job_id)


while [[ $status -eq "processing" ]]; do
    result=$(curl http://localhost:8000/api/jobs/${job_id} 2>/dev/null)
    status=$(echo $result | jq -r .status)
    percentage=$(echo $result | jq .progress.percentage)
    message=$(echo $result | jq .progress.message)
    echo "Current job ${status}: ${percentage}% complete (${message})"
    sleep 5
done

exit 0

# To delete all detections for video with ID 7:
# curl -X DELETE "http://localhost:8000/api/videos/7/detections"