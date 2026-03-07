#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash tools/batch_dist_test_epochs.sh --model-dir ./work_dirs/bevdet-r50
# Optional:
#   --gpus 4 --epochs "6 12 18 24"

RESULT_DIR="/home/dataset-local/lr/code/BEVDet/results"
WORK_DIR=""
NUM_GPU=4
EPOCHS_STR="6 12 18 24"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            WORK_DIR="${2:-}"
            shift 2
            ;;
        --gpus)
            NUM_GPU="${2:-4}"
            shift 2
            ;;
        --epochs)
            EPOCHS_STR="${2:-6 12 18 24}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model-dir <work_dir> [--gpus 4] [--epochs \"6 12 18 24\"]"
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1"
            echo "Usage: $0 --model-dir <work_dir> [--gpus 4] [--epochs \"6 12 18 24\"]"
            exit 1
            ;;
    esac
done

if [[ -z "${WORK_DIR}" ]]; then
    echo "Usage: $0 --model-dir <work_dir> [--gpus 4] [--epochs \"6 12 18 24\"]"
    exit 1
fi

if [[ ! -d "${WORK_DIR}" ]]; then
    echo "[ERROR] work_dir not found: ${WORK_DIR}"
    exit 1
fi

# In MMDet3D work_dirs, training config is usually dumped as a single .py file.
CONFIG=$(ls "${WORK_DIR}"/*.py 2>/dev/null | head -n 1 || true)
if [[ -z "${CONFIG}" ]]; then
    echo "[ERROR] no config .py found in: ${WORK_DIR}"
    exit 1
fi

mkdir -p "${RESULT_DIR}"

for epoch in ${EPOCHS_STR}; do
    checkpoint="${WORK_DIR}/epoch_${epoch}.pth"
    out_pkl="${RESULT_DIR}/epoch_${epoch}_result.pkl"
    out_log="${RESULT_DIR}/epoch_${epoch}_eval.log"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "[WARN] checkpoint not found, skip: ${checkpoint}" | tee -a "${RESULT_DIR}/missing_checkpoints.log"
        continue
    fi

    echo "[INFO] Testing ${checkpoint} with config ${CONFIG}"
    ./tools/dist_test.sh "${CONFIG}" "${checkpoint}" "${NUM_GPU}" \
        --out "${out_pkl}" \
        --eval mAP 2>&1 | tee "${out_log}"
done

echo "[DONE] All evaluations finished. Results are in: ${RESULT_DIR}"
