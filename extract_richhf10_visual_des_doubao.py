#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import base64
import mimetypes
from typing import Dict, Any, Optional, List

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from volcenginesdkarkruntime import Ark


# =========================================================
# Config (env vars)
# =========================================================
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_MODEL = os.environ.get("ARK_MODEL", "doubao-seed-1-6-251015")

IMAGE_DIR = os.environ.get("IMAGE_DIR", "./examples/richhf/images")
META_JSON = os.environ.get("META_JSON", "./examples/richhf/meta.json")
OUT_DIR = os.environ.get("OUT_DIR", "./examples/richhf/visual_descs")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "3"))
SLEEP_S = float(os.environ.get("SLEEP_S", "0.2"))

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "900"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))


# =========================================================
# Ark Client
# =========================================================
def make_client() -> Ark:
    if not ARK_API_KEY:
        raise RuntimeError("ARK_API_KEY is empty. Please export ARK_API_KEY first.")
    return Ark(base_url=ARK_BASE_URL, api_key=ARK_API_KEY)


client = make_client()


# =========================================================
# Prompt (your latest version, unchanged semantics)
# =========================================================
def system_prompt_richhf_visual_desc(prompt_text: str):
    return f"""
Task:
You are given an image and its intended text prompt.

Prompt:
\"{prompt_text}\"

Goal:
Output ONLY the image content that CONFLICTS with the prompt.

CRITICAL OUTPUT RULES:

1) EMPTY CASE:
- If the image reasonably matches the prompt with NO clear deviations, output ONLY: {{}}.

2) KEY GRANULARITY (VERY IMPORTANT):
- JSON keys MUST be short, minimal visual concept names.
- Each key should be 1–2 words whenever possible.
- Keys should represent a single object, entity, or attribute category.
- DO NOT include colors, shapes, positions, or multiple attributes in the key.
- DO NOT write full descriptive phrases as keys.
- JSON keys MUST be DETECTABLE visual entities.

GOOD keys:
- "bush"
- "round sign"
- "checkered cloth"
- "sofa"
- "metal body"
- "wood texture"

BAD keys:
- "green rounded bush with pink text"
- "pink and white checkered fabric surface"
- "background white couch with colorful pillows"

3) FEATURES:
For each key, output a list of all visual feature phrases including:
- basic type definition
- typical colors
- key shape or form
- notable components or materials

Output format (STRICT JSON ONLY):
{{
  "key1": ["feature1", "feature2", "..."],
  "key2": ["feature1", "feature2", "..."]
}}

Example:
Prompt: "a Ferrari car that is made out of wood"
Possible Output (if the image shows a metallic Ferrari):
{{
  "car body": [
    "sports car body",
    "glossy red color",
    "smooth reflective metal surface",
    "aerodynamic curved panels",
    "industrial metallic material"
  ],
  "wheels": [
    "silver alloy wheels",
    "black rubber tires",
    "low-profile tire shape",
    "multi-spoke wheel design",
    "performance-oriented size"
  ]
}}

Now analyze the given image and return ONLY the JSON object.
""".strip()



SYSTEM_RULE = (
    "You are an image anomaly detection expert. "
    "You must ONLY output valid JSON. "
    "No markdown, no explanations."
)


# =========================================================
# Image helpers: local file -> data URL
# =========================================================
def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext in [".png"]:
        return "image/png"
    if ext in [".webp"]:
        return "image/webp"
    return "application/octet-stream"


def file_to_data_url(path: str) -> str:
    mime = guess_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# =========================================================
# JSON cleanup + schema checks
# =========================================================
def clean_json_text(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()
    return s


def looks_bad_schema(d: Any) -> bool:
    if not isinstance(d, dict):
        return True
    bad_keys = {"category1", "category2", "object1", "object2", "concept1", "concept2"}
    keys = {str(k).strip().lower() for k in d.keys()}
    return any(k in bad_keys for k in keys)


# =========================================================
# Ark call helpers
# =========================================================
def extract_text_from_ark_response(resp: Any) -> str:
    """
    Ark responses API 返回结构可能随版本变化。
    这里用最稳的方式：尽量把 output 里所有 text 拼出来。
    """
    # 1) 直接转 dict 看看
    try:
        rdict = resp.to_dict()  # 某些版本支持
    except Exception:
        try:
            rdict = dict(resp)
        except Exception:
            rdict = None

    if isinstance(rdict, dict):
        # 常见：rdict["output"] 是 list，里面有 {"content":[{"type":"output_text","text":...}, ...]}
        out = rdict.get("output")
        if isinstance(out, list) and out:
            texts = []
            for item in out:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            texts.append(str(c["text"]))
            if texts:
                return "\n".join(texts).strip()

        # 兜底：有时直接在 rdict 里某个字段
        for k in ["text", "content", "result"]:
            if k in rdict and isinstance(rdict[k], str):
                return rdict[k].strip()

    # 2) 再试 attribute 访问（不同 SDK 可能是 resp.output[0].content[0].text）
    try:
        if hasattr(resp, "output") and resp.output:
            texts = []
            for item in resp.output:
                if hasattr(item, "content") and item.content:
                    for c in item.content:
                        if hasattr(c, "text"):
                            texts.append(str(c.text))
            if texts:
                return "\n".join(texts).strip()
    except Exception:
        pass

    # 3) 最后兜底：字符串化
    return str(resp)


def build_ark_input(prompt_text: str, image_data_url: str) -> List[Dict[str, Any]]:
    """
    Ark responses.create 的 input 格式：
    [
      {"role":"system","content":[{"type":"input_text","text":"..."}]},
      {"role":"user","content":[{"type":"input_image","image_url":"..."}, {"type":"input_text","text":"..."}]}
    ]
    """
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_RULE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_data_url},
                {"type": "input_text", "text": system_prompt_richhf_visual_desc(prompt_text)},
            ],
        },
    ]


def call_ark(input_payload: List[Dict[str, Any]], max_retries: int = MAX_RETRIES) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=ARK_MODEL,
                input=input_payload
            )
            return extract_text_from_ark_response(resp)
        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Ark API error: {e}")
            time.sleep(1.0)
    return None


# =========================================================
# Dataset
# =========================================================
def find_image_file(image_dir: str, base_name: str) -> str:
    name_without_ext = os.path.splitext(base_name)[0]
    exts = [".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"]

    p0 = os.path.join(image_dir, base_name)
    if os.path.exists(p0):
        return p0

    for ext in exts:
        p = os.path.join(image_dir, name_without_ext + ext)
        if os.path.exists(p):
            return p

    return p0


class RichHFMetaDataset(Dataset):
    def __init__(self, image_dir: str, meta_json: str):
        self.image_dir = image_dir
        with open(meta_json, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.ids = list(self.meta.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.ids[idx]
        item = self.meta[sid]

        filename = item.get("filename", "")
        img_name = os.path.basename(filename)
        img_path = find_image_file(self.image_dir, img_name)

        prompt = item.get("clean_prompt") or item.get("prompt") or ""

        return {"id": sid, "img_path": img_path, "prompt": prompt}

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        return {k: [b[k] for b in batch] for k in ["id", "img_path", "prompt"]}


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = RichHFMetaDataset(IMAGE_DIR, META_JSON)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ds.collate_fn)

    for batch in tqdm(dl):
        for sid, img_path, prompt in zip(batch["id"], batch["img_path"], batch["prompt"]):
            save_path = os.path.join(OUT_DIR, f"{sid}.json")
            if os.path.exists(save_path):
                continue

            if not os.path.exists(img_path):
                print(f"[Missing image] {img_path}")
                continue

            try:
                image_data_url = file_to_data_url(img_path)
            except Exception as e:
                print(f"[Bad image] {img_path}: {e}")
                continue

            ark_input = build_ark_input(prompt, image_data_url)
            raw = call_ark(ark_input)
            if raw is None:
                print(f"[Failed] id={sid}")
                continue

            try:
                txt = clean_json_text(raw)
                parsed = json.loads(txt)

                # placeholder keys -> one fix pass
                # if looks_bad_schema(parsed):
                #     fix_input = build_fix_input(prompt, txt, image_data_url)
                #     raw2 = call_ark(fix_input, max_retries=2)
                #     if raw2 is not None:
                #         txt2 = clean_json_text(raw2)
                #         parsed2 = json.loads(txt2)
                #         if not looks_bad_schema(parsed2):
                #             parsed = parsed2

                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[Parse Error] id={sid}: {e}")
                print("Raw output:", raw[:2000])

            time.sleep(SLEEP_S)

    print("Done.")


if __name__ == "__main__":
    main()
