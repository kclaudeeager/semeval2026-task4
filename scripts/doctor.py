"""Check the environment before a long run. Usage: python scripts/doctor.py"""
from __future__ import annotations
import importlib
import sys

REQUIRED = [("hydra", None), ("omegaconf", None), ("numpy", None),
            ("scipy", None), ("sklearn", None)]
EMBEDDINGS = [("torch", (2, 6)), ("transformers", (5, 0)), ("sentence_transformers", (5, 0))]
STRUCTURE = [("spacy", (3, 8)), ("maverick", None)]


def _version(mod) -> tuple[int, ...]:
    raw = getattr(mod, "__version__", "0")
    out = []
    for part in raw.split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out) or (0,)


def check(group: str, specs, hint: str) -> bool:
    print(f"\n{group}")
    ok = True
    for name, floor in specs:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"  MISSING  {name}")
            ok = False
            continue
        ver = getattr(mod, "__version__", "?")
        if floor and _version(mod)[:len(floor)] < floor:
            print(f"  TOO OLD  {name} {ver}  (need >= {'.'.join(map(str, floor))})")
            ok = False
        else:
            print(f"  ok       {name} {ver}")
    if not ok:
        print(f"  -> {hint}")
    return ok


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}  at  {sys.executable}")
    core = check("core", REQUIRED, 'uv pip install -e ".[dev]"')
    emb = check("L2 embeddings", EMBEDDINGS, 'uv pip install -e ".[embeddings]" --upgrade')
    st = check("L3/L4 structure", STRUCTURE,
               'uv pip install -e ".[structure]" && python -m spacy download en_core_web_sm')
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("  ok       en_core_web_sm")
    except Exception:
        print("  MISSING  en_core_web_sm -> python -m spacy download en_core_web_sm")
    print("\nrunnable layers:",
          "L0, L1" + (", L3n, L4n" if st else "") + (", L2, L3, L4, L5" if emb and st else ""))
    sys.exit(0 if core else 1)
