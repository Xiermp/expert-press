"""HuggingFace caches - inside the project folder, not on the system drive.

Import this module FIRST, before transformers/huggingface_hub:
    import hf_env  # noqa: F401

All downloads (GGUF from mradermacher, config/tokenizer from the base repo,
wikitext datasets) go to <project folder>/hf_cache/ - moved and deleted
together with the project; the system drive (C:\\Users\\...\\.cache) does not
grow.

If HF_HOME/HF_HUB_CACHE are set manually BEFORE the run, they are kept
(you can relocate the cache anywhere with your own values).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "hf_cache")


def apply() -> str:
    os.makedirs(CACHE, exist_ok=True)
    settings = {
        "HF_HOME": CACHE,
        "HF_HUB_CACHE": os.path.join(CACHE, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(CACHE, "hub"),  # legacy variable
        "HF_DATASETS_CACHE": os.path.join(CACHE, "datasets"),
    }
    changed = []
    for key, val in settings.items():
        if not os.environ.get(key):
            os.environ[key] = val
            changed.append(key)
    return ", ".join(changed)


if __name__ == "__main__":
    who = apply()
    print(f"hf_cache: {CACHE}")
    print(f"variables set: {who or 'none (already set outside)'}")
else:
    _changed = apply()
    if _changed and os.environ.get("MOE_QUIET_ENV") != "1":
        print(f"(HF cache: {CACHE})", flush=True)
