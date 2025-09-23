import yaml, csv, os, json

cfg = yaml.safe_load(open("config/config.yaml"))
marks = list(cfg["marks"]["core"])
if cfg.get("use_extra", False):
    marks += cfg["marks"].get("extra", [])
eids = cfg["eids"]

# 映射
eid2bio={}
with open("config/eid_to_biosample.tsv") as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        if r["EID"] and r["biosample_term_name"]:
            eid2bio[r["EID"].strip()] = r["biosample_term_name"].strip()

rule all:
    input: expand("hist/{eid}-{mark}.bw", eid=eids, mark=marks)

rule pick_url:
    output: "downloads/{eid}-{mark}.json"
    params: biosample=lambda w: eid2bio.get(w.eid, "")
    shell:
        r"""
        mkdir -p downloads hist
        if [ -z "{params.biosample}" ]; then
          echo '{{"found":false,"reason":"no biosample mapping"}}' > {output}
        else
          python scripts/encode_pick.py --biosample "{params.biosample}" --mark "{wildcards.mark}" > {output} || \
          echo '{{"found":false,"reason":"query error"}}' > {output}
        fi
        """

rule download_bw:
    input: "downloads/{eid}-{mark}.json"
    output: "hist/{eid}-{mark}.bw"
    run:
        meta=json.load(open(input[0]))
        if not meta.get("found", False):
            print(f"[WARN] Skip {wildcards.eid}-{wildcards.mark}: not found")
            shell(f': > "{output[0]}"')
        else:
            url=meta["url"]
            tmp=f"{output[0]}.part"
            shell(f'curl -L --retry 5 --retry-delay 5 -o "{tmp}" "{url}"')
            shell(f'mv "{tmp}" "{output[0]}"')