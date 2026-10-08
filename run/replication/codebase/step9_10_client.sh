#!/usr/bin/env bash
# Client run under with_judge.sh: policy uses GPUs 0,1; judge is on GPUs 2,3.
# 1) Step 10 eval: evaluate the 6 preventative-steer / CAFT checkpoints on all 3 traits.
# 2) Step 9: post-hoc negative-coef steering on the 3 misaligned_2 checkpoints.
set -u
export CUDA_VISIBLE_DEVICES=0,1
L=/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/logs

echo "########## STEP 10 EVAL (6 steer/ablate checkpoints) ##########"
CKPTS=$(python -c "import json; m=json.load(open('configs/generated/manifest.json')); print(' '.join(json.load(open(p))['output_dir'] for p in m['step10']))")
echo "step10 ckpts: $CKPTS"
python run_ckpt_eval.py --ckpts $CKPTS 2>&1 | tee $L/step10_eval.log | grep -i "done\|skip\|summary\|error\|traceback" | tail -40

echo "########## STEP 9 (post-hoc negative steering) ##########"
python run_step9.py 2>&1 | tee $L/step9.log | grep -i "done\|skip\|summary\|coef\|error\|traceback" | tail -60
