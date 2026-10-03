#!/usr/bin/env python3
"""Install a PREBUILT flash-attn wheel matching the installed torch.

Compiling flash-attn from source takes far longer than the 2h QoS allows, so we
resolve the correct wheel from the GitHub release assets instead of guessing a
filename. Fails loudly if no wheel matches rather than silently
leaving the model without FA2.
"""
import json
import subprocess
import sys
import urllib.request

import torch

API = "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=30"


def main():
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    needle = f"torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
    print(f"[fa] torch={torch.__version__} abi={abi} py={py}")
    print(f"[fa] looking for assets containing: ...{needle}")

    req = urllib.request.Request(API, headers={"User-Agent": "j1-setup"})
    with urllib.request.urlopen(req, timeout=60) as r:
        releases = json.load(r)

    cands = []
    for rel in releases:
        for a in rel.get("assets", []):
            n = a["name"]
            if n.endswith(needle) and "cu12" in n:
                cands.append((rel.get("tag_name", ""), n, a["browser_download_url"]))
    if not cands:
        print("[fa] AVAILABLE sample asset names (first 15):")
        seen = 0
        for rel in releases:
            for a in rel.get("assets", []):
                print("   ", a["name"])
                seen += 1
                if seen >= 15:
                    break
            if seen >= 15:
                break
        sys.exit("[fa] FAILURE: no prebuilt flash-attn wheel matches this torch/python/ABI")

    cands.sort(key=lambda t: t[0], reverse=True)
    tag, name, url = cands[0]
    print(f"[fa] selected {tag} :: {name}")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "--no-deps", url])
    if rc != 0:
        sys.exit(f"[fa] FAILURE: pip install of {url} returned {rc}")

    import importlib
    fa = importlib.import_module("flash_attn")
    print(f"[fa] installed flash_attn {getattr(fa, '__version__', '?')}")


if __name__ == "__main__":
    main()
