#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "${PROJECT_ROOT}/training/dpo/scripts/repair_corrupted_data.sh" "$@"

