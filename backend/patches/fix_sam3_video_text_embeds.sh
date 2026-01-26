#!/bin/bash
# Fix for CLIPTextModelOutput bug in SAM3 video model
# Issue: get_text_features() returns BaseModelOutputWithPooling, but the code
# expects a tensor. Need to extract .pooler_output
#
# The bug is in transformers main branch in models/sam3_video/modeling_sam3_video.py
# Line ~593: needs .pooler_output after get_text_features() call

set -e

# Find the transformers installation
TRANSFORMERS_PATH=$(python -c "import transformers; print(transformers.__path__[0])")

if [ -z "$TRANSFORMERS_PATH" ]; then
    echo "ERROR: transformers not installed"
    exit 1
fi

TARGET_FILE="$TRANSFORMERS_PATH/models/sam3_video/modeling_sam3_video.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "WARNING: $TARGET_FILE not found. SAM3 video model may not be installed."
    exit 0
fi

# Check if already patched
if grep -q "\.pooler_output  # Extract tensor from CLIPTextModelOutput" "$TARGET_FILE"; then
    echo "SAM3 video text_embeds fix already applied"
    exit 0
fi

# Apply the fix using sed
# Find the line with just "                )" after the get_text_features call
# and replace it with "                ).pooler_output  # Extract tensor from CLIPTextModelOutput"
# We need to be specific - the closing paren that's 16 spaces indented after the attention_mask line
sed -i '/attention_mask=inference_session\.prompt_attention_masks\[prompt_id\],/{n;s/^                )$/                ).pooler_output  # Extract tensor from CLIPTextModelOutput/}' "$TARGET_FILE"

# Verify the fix was applied
if grep -q "\.pooler_output  # Extract tensor from CLIPTextModelOutput" "$TARGET_FILE"; then
    echo "SAM3 video text_embeds fix applied successfully"
else
    echo "INFO: Patch pattern not found - bug may be fixed upstream in transformers"
    echo "Continuing without patch (will fail at runtime if bug still exists)"
fi
