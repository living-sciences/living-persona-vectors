import json, sys
LOG="/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/replication_log.json"
entry=json.load(open(sys.argv[1]))
d=json.load(open(LOG))
# replace if step_id exists else append
d["step_outcomes"]=[s for s in d["step_outcomes"] if s["step_id"]!=entry["step_id"]]
d["step_outcomes"].append(entry)
d["step_outcomes"].sort(key=lambda s:s["step_id"])
json.dump(d,open(LOG,"w"),indent=4)
print("logged step",entry["step_id"],"; total steps:",len(d["step_outcomes"]))
