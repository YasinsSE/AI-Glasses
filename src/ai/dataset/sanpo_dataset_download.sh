#!/bin/bash

# --- USER CONFIGURATION ---
# Number of valid (masked) sessions to skip (previously downloaded)
SKIP_COUNT=75
#SKIP_COUNT=1
#TARGET_COUNT=1
# Target number of NEW valid (masked) sessions to download
TARGET_COUNT=15

# Destination Directory
DEST_DIR="./sanpo_dataset"
mkdir -p "$DEST_DIR"

# Google Cloud Paths
BASE_URL="gs://gresearch/sanpo_dataset/v0/sanpo-real"
SPLITS_URL="gs://gresearch/sanpo_dataset/v0/sanpo-real/splits/train_session_ids.txt"

# --- EXCLUDE PATTERNS ---
# Strictly ignoring 'right' camera folders and depth maps
EXCLUDE_PATTERN=".*right/.*|.*depth_maps/.*|.*zed_depth_maps/.*"

echo "=== ALAS Project: Smart Data Miner (Auto-Miner) v3 Initiated ==="
echo ">> Valid sessions to skip: $SKIP_COUNT"
echo ">> Target new valid sessions: $TARGET_COUNT"

# Step 1: Download Label Map
echo "[1/3] Fetching Label Map (labelmap.json)..."
gsutil cp gs://gresearch/sanpo_dataset/v0/labelmap.json "$DEST_DIR/" >/dev/null 2>&1

# Step 2: Fetch Candidate List
echo "[2/3] Updating training list (train_session_ids.txt)..."
gsutil cp "$SPLITS_URL" "$DEST_DIR/all_candidates.txt" >/dev/null 2>&1

# Read the list into an array
CANDIDATE_SESSIONS=($(cat "$DEST_DIR/all_candidates.txt"))
TOTAL_CANDIDATES=${#CANDIDATE_SESSIONS[@]}

echo ">> Total Candidates: $TOTAL_CANDIDATES"
echo "[3/3] Scanning and Downloading Process Started..."

# Step 3: Smart Loop
downloaded_count=0
checked_count=0
skipped_valid_count=0

for SESSION_ID in "${CANDIDATE_SESSIONS[@]}"; do
    # Terminate if target count is reached
    if [ "$downloaded_count" -ge "$TARGET_COUNT" ]; then
        echo "🎉 TARGET REACHED! Successfully downloaded $TARGET_COUNT new masked sessions."
        break
    fi

    ((checked_count++))
    echo "----------------------------------------------------------------"
    echo "[$checked_count/$TOTAL_CANDIDATES] Checking session: $SESSION_ID"

    # Paths updated for chest camera
    CLOUD_SESSION="$BASE_URL/$SESSION_ID"
    CLOUD_CHEST="$CLOUD_SESSION/camera_chest"
    MASK_CHECK_PATH="$CLOUD_CHEST/left/segmentation_masks"

    # --- MASK VALIDATION ---
    if gsutil ls "$MASK_CHECK_PATH/" >/dev/null 2>&1; then
        
        # Check if this valid session should be skipped
        if [ "$skipped_valid_count" -lt "$SKIP_COUNT" ]; then
            ((skipped_valid_count++))
            echo "   ⏭️ SKIPPED: Valid masks exist, but marked for skipping. (Skipped: $skipped_valid_count/$SKIP_COUNT)"
            continue
        fi

        echo "   ✅ APPROVED: New valid masks detected. Initializing download..."
        
        # Prepare Local Directories
        LOCAL_SESSION="$DEST_DIR/$SESSION_ID"
        mkdir -p "$LOCAL_SESSION/camera_chest"

        # A) Description JSON
        gsutil cp "$CLOUD_SESSION/description.json" "$LOCAL_SESSION/" 2>/dev/null

        # B) Camera Chest (Includes everything except exclude pattern)
        gsutil -m -o "GSUtil:parallel_process_count=1" rsync -r \
          -x "$EXCLUDE_PATTERN" \
          "$CLOUD_CHEST" \
          "$LOCAL_SESSION/camera_chest/"

        ((downloaded_count++))
        echo "   💾 DOWNLOAD COMPLETE. (Total New Downloads: $downloaded_count/$TARGET_COUNT)"
        
    else
        echo "   ❌ BYPASSED: No valid masks found in this session's chest camera."
    fi

done

if [ "$downloaded_count" -lt "$TARGET_COUNT" ]; then
    echo "WARNING: Candidate list exhausted. Only found $downloaded_count suitable sessions."
else
    echo "=== OPERATION SUCCESSFULLY COMPLETED ==="
    echo "Dataset located at: $(pwd)/$DEST_DIR"
fi