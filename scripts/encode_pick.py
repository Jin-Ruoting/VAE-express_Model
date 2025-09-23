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

def enc_query(params):
    base = "https://www.encodeproject.org/search/"
    params.setdefault("type", "File")
    params.setdefault("file_format", "bigWig")
    params.setdefault("assembly", "GRCh38")
    params.setdefault("status", "released")
    params.setdefault("format", "json")
    params.setdefault("limit", "all")
    params["field"] = ["accession","href","target.label","biosample_ontology.term_name",
                       "assembly","output_type","assay_title","preferred_default"]
    q=[]
    for k,v in params.items():
        if isinstance(v, list):
            for vi in v: q.append((k,vi))
        else:
            q.append((k,v))
    url = base + "?" + urllib.parse.urlencode(q)
    r = requests.get(url, headers={"accept":"application/json"})
    r.raise_for_status()
    return r.json().get("@graph", [])

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
    for ot in OUTPUT_TYPES:
        params = {
            "assay_title": assay,
            "biosample_ontology.term_name": args.biosample,
            "assembly": "GRCh38",
            "output_type": ot,
        }
        if target:
            params["target.label"] = target
        files = enc_query(params)
        best = pick_best(files)
        if best:
            found = best
            used_output = ot
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