import os, json, time, uuid
from typing import Dict, List, Optional
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INDEX_FILE = os.path.join(OUTPUT_DIR, "generations.jsonl")

def _ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

def save_run(run_id: str, meta: Dict, images: List[Image.Image], control_preview: Optional[Image.Image] = None) -> str:
    _ensure_dirs()
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # save images
    img_paths = []
    for i, img in enumerate(images, 1):
        p = os.path.join(run_dir, f"image_{i}.png")
        img.save(p)
        img_paths.append(p)

    control_path = None
    if control_preview is not None:
        control_path = os.path.join(run_dir, "control_preview.png")
        control_preview.save(control_path)

    meta2 = dict(meta)
    meta2["run_id"] = run_id
    meta2["run_dir"] = run_dir
    meta2["image_paths"] = img_paths
    meta2["control_preview_path"] = control_path

    # write meta.json inside run folder
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta2, f, indent=2)

    # append to global index
    with open(INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "timestamp": meta2.get("timestamp"),
            "mode": meta2.get("mode"),
            "style": meta2.get("agent", {}).get("style"),
            "goal": meta2.get("agent", {}).get("goal") or meta2.get("goal"),
            "run_dir": run_dir,
        }, ensure_ascii=False) + "\n")

    return run_dir

def load_index(limit: int = 50) -> List[Dict]:
    _ensure_dirs()
    if not os.path.exists(INDEX_FILE):
        return []

    rows = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except:
                pass

    # newest first
    rows = rows[::-1]
    return rows[:limit]

def load_run_meta(run_id: str) -> Optional[Dict]:
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
