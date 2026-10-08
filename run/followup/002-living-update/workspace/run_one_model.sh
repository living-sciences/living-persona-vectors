#!/usr/bin/env bash
# Followup 001 — full per-model pipeline (foreground). Usage:
#   bash run_one_model.sh <SLUG> <MODEL_PATH> <COARSE_LAYERS_CSV>
# e.g. bash run_one_model.sh Llama-3.1-8B-Instruct /net/.../Llama-3.1-8B-Instruct 14,18,21,24
set -u
SLUG="$1"; MODELP="$2"; COARSE="$3"
STAGE="${STAGE:-all}"   # 1 = extract..pick_layers ; 2 = confirm..rejudge2 ; all = both
source "$(dirname "$0")/env.sh"
cd "$CB"
PY=$VENV/bin/python
LOG=$FU/results/logs/$SLUG
mkdir -p "$LOG"
export PV_MODEL="$MODELP"
export PV_SLUG="$SLUG"
export PV_EXTRACT_DIR="eval_persona_extract/$SLUG"
export PV_VECDIR="persona_vectors/$SLUG"
export PV_OUTDIR="eval_persona_eval/$SLUG"
ALLTRAITS="evil,sycophantic,hallucinating"

echo "############## MODEL $SLUG  ($MODELP)  STAGE=$STAGE ##############"
date

if [ "$STAGE" = "2" ]; then
  ACTIVE=$(cat "$LOG/active_traits.txt")
  export PV_SEL_LAYER=$(cat "$FU/results/logs/$SLUG/sel_layer.json")
  echo "resuming STAGE 2 with ACTIVE='$ACTIVE' SEL=$PV_SEL_LAYER"
fi

if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
# ---- Step 1a: extraction GENERATION only (judging disabled; vLLM + in-process
#      OpenRouter judging are incompatible, so we judge in a separate clean process). ----
echo ">>> [1a] extract (generation, no inline judge)"; export PV_NO_JUDGE=1
PV_N_PER_Q=10 taskset -c 0-15 $PY run_extract.py 2>&1 | tee "$LOG/1_extract.log"
# ---- Step 1a2: judge extraction CSVs (paper-exact gpt-4.1-mini, separate process) ----
echo ">>> [1a2] judge extraction pairs"; unset PV_NO_JUDGE
taskset -c 0-15 $PY $FU/workspace/judge_extract.py "$PV_EXTRACT_DIR" evil sycophantic hallucinating 2>&1 | tee "$LOG/1_judge_extract.log"

# ---- Step 1b: genvec (+ effective-pair data-hole check) ----
echo ">>> [1b] genvec"
taskset -c 0-15 $PY run_genvec.py 2>&1 | tee "$LOG/1_genvec.log"

# Determine ACTIVE traits (effective pairs >= 30) from genvec log
$PY - "$LOG/1_genvec.log" <<'PY' > "$LOG/active_traits.txt"
import re,sys
log=open(sys.argv[1]).read()
active=[]
for m in re.finditer(r"=== (\w+): effective pairs = (\d+)", log):
    t,n=m.group(1),int(m.group(2))
    if n>=30: active.append(t)
print(",".join(active))
PY
ACTIVE=$(cat "$LOG/active_traits.txt")
echo "ACTIVE TRAITS (>=30 effective pairs): '$ACTIVE'"
if [ -z "$ACTIVE" ]; then echo "NO ACTIVE TRAITS — aborting model $SLUG"; exit 2; fi

# ---- Step 2: baseline (all 3 traits; vector-independent). No inline judge. ----
echo ">>> [2] baseline"; export PV_NO_JUDGE=1
PV_TRAITS="$ALLTRAITS" PV_N_PER_Q=10 taskset -c 0-15 $PY run_baseline.py 2>&1 | tee "$LOG/2_baseline.log"

# ---- Step 3a: steering COARSE scan (active traits). No inline judge. ----
echo ">>> [3a] steer coarse (layers $COARSE)"
PV_LAYERS="$COARSE" PV_COEFS="1.0,1.5,2.0" PV_N_PER_Q=5 PV_MAX_TOKENS=512 \
  taskset -c 0-15 $PY run_steer_sweep.py --traits ${ACTIVE//,/ } 2>&1 | tee "$LOG/3_steer_coarse.log"

# ---- Rejudge phase 1: baseline + coarse (paper-exact gpt-4.1-mini) ----
echo ">>> [rejudge-1] baseline + coarse"
unset PV_NO_JUDGE
REJ_GLOB="$CB/eval_persona_eval/$SLUG/*.csv" taskset -c 0-15 $PY $FU/workspace/rejudge_all.py 2>&1 | tee "$LOG/4_rejudge1.log"

# ---- Pick selected layer per active trait = argmax gpt trait_mean over coarse grid ----
$PY $FU/workspace/pick_layers.py "$SLUG" "$COARSE" "$ACTIVE" 2>&1 | tee "$LOG/5_pick_layers.log"
SEL=$(cat "$FU/results/logs/$SLUG/sel_layer.json")
echo "SELECTED LAYERS: $SEL"
export PV_SEL_LAYER="$SEL"
fi  # end STAGE 1

if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
# ---- Step 3b: steering CONFIRM sweep at selected layer (active traits) ----
echo ">>> [3b] steer confirm"; export PV_NO_JUDGE=1
# One trait at a time so each uses its own selected layer.
for t in ${ACTIVE//,/ }; do
  L=$($PY -c "import json,sys;print(json.load(open('$FU/results/logs/$SLUG/sel_layer.json'))['$t'])")
  echo "   confirm $t at layer $L"
  PV_OUTDIR="eval_persona_eval/$SLUG/confirm" PV_LAYERS="$L" PV_COEFS="0.5,1.0,1.5,2.0,2.5" PV_N_PER_Q=10 PV_MAX_TOKENS=512 \
    taskset -c 0-15 $PY run_steer_sweep.py --traits $t 2>&1 | tee -a "$LOG/6_steer_confirm.log"
done

# ---- Step 4: monitor generation + projection (active traits) ----
echo ">>> [4] monitor gen"
PV_TRAITS="$ACTIVE" PV_N_PER_Q=5 taskset -c 0-15 $PY run_monitor.py 2>&1 | tee "$LOG/7_monitor.log"
echo ">>> [4] monitor projection"
export PV_ANALYSIS="$FU/results/monitor_proj/$SLUG"; mkdir -p "$PV_ANALYSIS"
taskset -c 0-15 $PY run_monitor_proj.py 2>&1 | tee "$LOG/8_monitor_proj.log"

# ---- Step 5: sample separability (active traits) ----
echo ">>> [5] step8 separability"
export PV_ANALYSIS="$FU/results/step8/$SLUG"; mkdir -p "$PV_ANALYSIS"
taskset -c 0-15 $PY run_step8.py 2>&1 | tee "$LOG/9_step8.log"

# ---- Rejudge phase 2: confirm + monitor ----
echo ">>> [rejudge-2] confirm + monitor"; unset PV_NO_JUDGE
REJ_GLOB="$CB/eval_persona_eval/$SLUG/**/*.csv" taskset -c 0-15 $PY $FU/workspace/rejudge_all.py 2>&1 | tee "$LOG/10_rejudge2.log"
fi  # end STAGE 2

echo "############## DONE $SLUG (stage $STAGE) ##############"; date
