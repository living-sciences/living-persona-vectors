# Followup 004 shared environment. `source` this before any step.
export HF_HOME=/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OPENAI_API_KEY=$(grep '^OPENROUTER_API_KEY=' "${OPENROUTER_ENV_FILE:-$HOME/.openrouter.env}" | head -1 | cut -d= -f2-)
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export HF_TOKEN=dummy_not_needed_local_paths
export FU=/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/004-living-update
export REJ_CB=$FU/workspace/codebase
export REJ_BASE=$FU/results
export CB=$FU/workspace/codebase
export VENV=$FU/workspace/.venv-tf
