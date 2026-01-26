#!/bin/bash
# Fix for torch.compile compatibility with SAM3 tracker model
# Issue: x.permute().view() creates non-contiguous tensor that inductor can't handle
# Fix: Use .reshape() instead of .view() (reshape handles non-contiguous tensors)
#
# The bug is in transformers models/sam3_tracker_video/modeling_sam3_tracker_video.py
# Line ~2595: x.permute(1, 2, 0).view(...) -> x.permute(1, 2, 0).reshape(...)

set -e

# Find the transformers installation
TRANSFORMERS_PATH=$(python -c "import transformers; print(transformers.__path__[0])")

if [ -z "$TRANSFORMERS_PATH" ]; then
    echo "ERROR: transformers not installed"
    exit 1
fi

TARGET_FILE="$TRANSFORMERS_PATH/models/sam3_tracker_video/modeling_sam3_tracker_video.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "WARNING: $TARGET_FILE not found. SAM3 tracker model may not be installed."
    exit 0
fi

# Check if already patched
if grep -q "\.permute(1, 2, 0)\.reshape(" "$TARGET_FILE"; then
    echo "SAM3 torch.compile fix already applied"
    exit 0
fi

# Apply the fix - replace .view( with .reshape( after permute(1, 2, 0)
# This handles non-contiguous tensors that torch.compile can't optimize with .view()
sed -i 's/\.permute(1, 2, 0)\.view(/.permute(1, 2, 0).reshape(/g' "$TARGET_FILE"

# Verify the fix was applied
if grep -q "\.permute(1, 2, 0)\.reshape(" "$TARGET_FILE"; then
    echo "SAM3 torch.compile fix applied successfully"
else
    echo "WARNING: Failed to apply SAM3 torch.compile fix"
    echo "Pattern may have changed in transformers - check manually"
    exit 0
fi
