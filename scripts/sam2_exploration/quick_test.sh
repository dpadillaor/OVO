#!/bin/bash
# Quick test sequences for common parameter explorations

cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_test() {
    local desc=$1
    local params=$2

    echo -e "${YELLOW}=== $desc ===${NC}"
    python test_sam2.py $params --verbose
    echo ""
}

# Test 1: Points per side sensitivity
echo -e "${GREEN}EXPERIMENT 1: Points per side (sampling density)${NC}"
run_test "Sparse (8 points)" "--points-per-side 8 --frames 0"
run_test "Default (16 points)" "--points-per-side 16 --frames 0"
run_test "Dense (32 points)" "--points-per-side 32 --frames 0"

# Test 2: NMS IoU threshold sensitivity
echo -e "${GREEN}EXPERIMENT 2: NMS IoU threshold (mask overlap tolerance)${NC}"
run_test "Strict (IoU < 0.3)" "--nms-iou-th 0.3 --frames 0"
run_test "Default (IoU < 0.8)" "--nms-iou-th 0.8 --frames 0"
run_test "Relaxed (IoU < 0.9)" "--nms-iou-th 0.9 --frames 0"

# Test 3: Combined effect
echo -e "${GREEN}EXPERIMENT 3: Combined effects${NC}"
run_test "Fast: sparse + relaxed NMS" "--points-per-side 8 --nms-iou-th 0.9 --frames 0"
run_test "Balanced: default" "--points-per-side 16 --nms-iou-th 0.8 --frames 0"
run_test "Quality: dense + strict NMS" "--points-per-side 32 --nms-iou-th 0.3 --frames 0"

echo -e "${GREEN}All tests complete. Results in: output/${NC}"
echo "Compare with: python visualize_results.py --dataset data/input/Datasets/replica/office_0/rgb"
