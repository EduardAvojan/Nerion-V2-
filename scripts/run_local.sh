set -euo pipefail
[ -f .env ] && export $(grep -v ^# .env | xargs)
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
exec python -m app.nerion_chat "$@"
