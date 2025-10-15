import json, yaml, sys, shutil
from pathlib import Path

def main():
    cfg = yaml.safe_load(open('config/config.yaml'))
    eids = cfg['eids']
    marks = list(cfg['marks']['core']) + (cfg['marks'].get('extra',[]) if cfg.get('use_extra') else [])
    stats_path = Path(cfg['paths']['stats_json'])
    if not stats_path.exists():
        sys.exit(f"[ERR] not found: {stats_path}")

    stats = json.load(open(stats_path))
    keys = list(stats.keys())
    print(f"[INFO] loaded {len(keys)} top-level keys; example: {keys[:5]}")

    out = {}

    # 形态A：已经是 EID:MARK
    if any(':' in k for k in keys):
        print("[INFO] Detected 'EID:MARK' keys; normalizing and filling missing if possible.")
        for e in eids:
            for m in marks:
                k = f"{e}:{m}"
                if k in stats:
                    out[k] = stats[k]
                elif m in stats:
                    out[k] = stats[m]
                elif f"{e}-{m}" in stats:
                    out[k] = stats[f"{e}-{m}"]
                else:
                    print(f"[WARN] missing stat for {k}")

    # 形态B：EID-MARK（连字符）
    elif any('-' in k for k in keys):
        print("[INFO] Detected 'EID-MARK' keys; converting to 'EID:MARK'.")
        for e in eids:
            for m in marks:
                k_dash = f"{e}-{m}"
                k_col = f"{e}:{m}"
                if k_dash in stats:
                    out[k_col] = stats[k_dash]
                elif m in stats:
                    out[k_col] = stats[m]
                else:
                    print(f"[WARN] missing stat for {k_dash}")

    # 形态C：顶层为 EID，value 是以 mark 为键的子字典（当前情况）
    elif all(isinstance(stats[k], dict) for k in keys) and set(marks) & set(stats[keys[0]].keys()):
        print("[INFO] Detected nested dict 'EID -> MARK -> {q1,q99}'; flattening.")
        for e in eids:
            sub = stats.get(e, {})
            if not isinstance(sub, dict):
                print(f"[WARN] {e} is not a dict, skip"); continue
            for m in marks:
                if m in sub:
                    out[f"{e}:{m}"] = sub[m]
                else:
                    print(f"[WARN] missing stat for {e}:{m}")

    # 形态D：仅有全局 mark 层
    elif set(keys) >= set(marks):
        print("[INFO] Detected mark-level stats; expanding to each EID:MARK.")
        for e in eids:
            for m in marks:
                if m in stats:
                    out[f"{e}:{m}"] = stats[m]
                else:
                    print(f"[WARN] missing global stat for {m}")

    else:
        print("[ERR] Unrecognized stats format. Keys example:", keys[:10])
        sys.exit(2)

    expect = len(eids) * len(marks)
    have = len(out)
    print(f"[INFO] will write {have}/{expect} EID:MARK entries")

    # 备份并写回
    bak = stats_path.with_suffix(stats_path.suffix + ".bak")
    shutil.copy2(stats_path, bak)
    with open(stats_path, "w") as f:
        json.dump(out, f)
    print(f"[OK] saved normalized stats to {stats_path} (backup -> {bak})")

    # 验证键完整性
    miss = []
    for e in eids:
        for m in marks:
            if f"{e}:{m}" not in out:
                miss.append(f"{e}:{m}")
    print(f"[CHECK] missing after fix: {len(miss)}")
    if miss:
        print("[CHECK] example:", miss[:10])

if __name__ == "__main__":
    main()