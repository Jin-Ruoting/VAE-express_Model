import argparse, urllib.parse, requests, json, sys

MARK2Q = {
    "H3K4me3": ("Histone ChIP-seq", "H3K4me3"),
    "H3K27ac": ("Histone ChIP-seq", "H3K27ac"),
    "H3K4me1": ("Histone ChIP-seq", "H3K4me1"),
    "H3K36me3": ("Histone ChIP-seq", "H3K36me3"),
    "H3K27me3": ("Histone ChIP-seq", "H3K27me3"),
    "H3K9me3": ("Histone ChIP-seq", "H3K9me3"),
    "H3K9ac":  ("Histone ChIP-seq", "H3K9ac"),
    "ATAC":    ("ATAC-seq", None),
    "POLR2A":  ("TF ChIP-seq", "POLR2A"),
}

OUTPUT_TYPES = ["signal p-value", "fold change over control", "signal"]

HEADERS = {
    "accept": "application/json",
    "User-Agent": "Mozilla/5.0 (encode-script)"
}

def enc_get(params):
    base = "https://www.encodeproject.org/search/"
    # 基础参数
    base_params = {
        "type": "File",
        "file_format": "bigWig",
        "assembly": "GRCh38",
        "status": "released",
        "format": "json",
        "limit": "all",
    }
    field_list = ["accession","href","target.label","biosample_ontology.term_name",
                  "assembly","output_type","assay_title","preferred_default"]
    q=[]
    for k,v in {**base_params, **params}.items():
        if isinstance(v, list):
            for vi in v: q.append((k,vi))
        else:
            q.append((k,v))
    # 重复 field 参数
    for f in field_list:
        q.append(("field", f))
    url = base + "?" + urllib.parse.urlencode(q)
    r = requests.get(url, headers=HEADERS, timeout=30)
    # 有些组合返回 404，这里不抛异常，交由上层降级处理
    if r.status_code != 200:
        return r.status_code, []
    try:
        data = r.json()
    except Exception:
        return r.status_code, []
    return r.status_code, data.get("@graph", [])

def pick_best(files):
    pref=[f for f in files if f.get("preferred_default") is True]
    return pref[0] if pref else (files[0] if files else None)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--mark", required=True)
    args = ap.parse_args()

    if args.mark not in MARK2Q:
        print(json.dumps({"found": False, "reason": "unknown mark"})); sys.exit(0)

    assay, target = MARK2Q[args.mark]
    found = None
    used_output = None

    # 1) 严格查询（term_name 精确匹配）
    for ot in OUTPUT_TYPES:
        strict = {
            "assay_title": assay,
            "output_type": ot,
            "biosample_ontology.term_name": args.biosample,
        }
        if target: strict["target.label"] = target
        code, files = enc_get(strict)
        best = pick_best(files)
        if best:
            found, used_output = best, ot
            break
        # 2) 降级查询：改用 searchTerm 模糊匹配 biosample
        relaxed = {
            "assay_title": assay,
            "output_type": ot,
            "searchTerm": args.biosample,
        }
        if target: relaxed["target.label"] = target
        code2, files2 = enc_get(relaxed)
        best2 = pick_best(files2)
        if best2:
            found, used_output = best2, ot
            break

    if not found:
        print(json.dumps({"found": False})); sys.exit(0)

    print(json.dumps({
        "found": True,
        "accession": found["accession"],
        "url": "https://www.encodeproject.org" + found["href"],
        "assembly": found.get("assembly"),
        "assay_title": found.get("assay_title"),
        "output_type": used_output
    }))