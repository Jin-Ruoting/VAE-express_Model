import csv, subprocess, json, sys

tsv = "config/eid_to_biosample.tsv"
mark = "H3K27ac"  # 用一个常见标记做探针

ok, fail = [], []
with open(tsv) as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        eid = r["EID"].strip(); bio = r["biosample_term_name"].strip()
        if not eid or not bio: continue
        try:
            out = subprocess.check_output(
                ["python", "scripts/encode_pick.py", "--biosample", bio, "--mark", mark],
                text=True
            )
            meta = json.loads(out)
            if meta.get("found"):
                ok.append((eid, bio, meta.get("accession")))
            else:
                fail.append((eid, bio))
        except Exception as e:
            fail.append((eid, bio))

print("[OK]", len(ok), "biosamples resolved for", mark)
for x in ok: print("  ", x)
print("[FAIL]", len(fail), "biosamples unresolved")
for x in fail: print("  ", x)