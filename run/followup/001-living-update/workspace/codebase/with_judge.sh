#!/usr/bin/env bash
# Start the local vLLM judge (Qwen2.5-32B-Instruct) on GPU 3, wait until it is
# ready, run the passed client command (foreground), then tear the judge down.
# Everything runs inside ONE shell so the background judge lives for the whole
# call (this environment kills processes that outlive their launching Bash call).
#
# Usage: bash with_judge.sh <client command and args...>
set -u
LOGDIR=/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/logs
JUDGE_GPU=${JUDGE_GPU:-2,3}
PORT=${JUDGE_PORT:-8100}
# tensor-parallel size = number of comma-separated devices in JUDGE_GPU
JUDGE_TP=$(awk -F, '{print NF}' <<< "$JUDGE_GPU")

cleanup() {
  echo "[with_judge] stopping judge (pid $JPID)"
  kill $JPID 2>/dev/null
  wait $JPID 2>/dev/null
}
trap cleanup EXIT

echo "[with_judge] starting judge on GPU $JUDGE_GPU (TP=$JUDGE_TP) port $PORT"
CUDA_VISIBLE_DEVICES=$JUDGE_GPU vllm serve Qwen/Qwen2.5-32B-Instruct \
    --port $PORT --max-model-len 8192 --tensor-parallel-size $JUDGE_TP > $LOGDIR/judge_server.log 2>&1 &
JPID=$!

echo "[with_judge] waiting for judge readiness (pid $JPID)..."
for i in $(seq 1 120); do  # up to ~20 min
  if ! kill -0 $JPID 2>/dev/null; then
    echo "[with_judge] judge process died during startup; see judge_server.log"; exit 1
  fi
  if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "Qwen2.5-32B"; then
    echo "[with_judge] judge ready after ${i}0s"
    break
  fi
  sleep 10
done
if ! curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "Qwen2.5-32B"; then
  echo "[with_judge] judge never became ready"; exit 1
fi

echo "[with_judge] running client: $*"
"$@"
RC=$?
echo "[with_judge] client exited rc=$RC"
exit $RC
