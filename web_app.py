# -*- coding: utf-8 -*-
"""FastAPI backend — Novel Manager v3 (with settings support)"""
import json, os, uuid, datetime, importlib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
NOVELS_DIR = HERE / "novels"
NOVELS_DIR.mkdir(exist_ok=True)

app = FastAPI()


# ── settings from request headers ──

def _apply_settings(headers):
    """Read custom headers and set env vars before reloading pipeline."""
    api_key = headers.get("x-api-key") or headers.get("x-apikey") or ""
    model = headers.get("x-model") or ""
    base_url = headers.get("x-base-url") or headers.get("x-baseurl") or ""

    if api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        os.environ["NOVEL_MOCK"] = "0"
    else:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["NOVEL_MOCK"] = "1"

    if model.strip():
        os.environ["NOVEL_LLM_MODEL"] = model.strip()

    if base_url.strip():
        os.environ["OPENAI_API_BASE"] = base_url.strip()
        os.environ["OPENAI_BASE_URL"] = base_url.strip()
    else:
        os.environ.pop("OPENAI_API_BASE", None)
        os.environ.pop("OPENAI_BASE_URL", None)


def _reload():
    import novel_pipeline as np
    importlib.reload(np)
    if np.llm is None and not np.MOCK:
        os.environ["NOVEL_MOCK"] = "1"
        importlib.reload(np)
    return np


def _load(nid):
    p = NOVELS_DIR / f"{nid}.json"
    if not p.exists():
        raise HTTPException(404, "不存在")
    return json.loads(p.read_text("utf-8"))


def _save(nid, data):
    (NOVELS_DIR / f"{nid}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class CreateReq(BaseModel):
    title: str = ""
    idea: str = ""

class SaveReq(BaseModel):
    id: str = ""
    title: str = ""
    idea: str = ""
    world_setting: dict = {}
    characters: list = []
    outline: list = []
    chapters: list = []
    created_at: str = ""


@app.get("/")
def index():
    return HTMLResponse((HERE / "templates" / "index.html").read_text("utf-8"))


@app.get("/api/novels")
def list_novels():
    items = []
    for f in sorted(NOVELS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        d = json.loads(f.read_text("utf-8"))
        items.append({
            "id": d["id"], "title": d["title"],
            "created_at": d["created_at"],
            "idea_preview": d.get("idea", "")[:80],
            "chapter_count": len(d.get("chapters", [])),
        })
    return items


@app.post("/api/novels", status_code=201)
async def create_novel(body: CreateReq, request: Request):
    _apply_settings(request.headers)
    if not body.idea.strip():
        raise HTTPException(400, "创意不能为空")
    np = _reload()
    out = np.generate_outline({"user_idea": body.idea})
    novel = {
        "id": uuid.uuid4().hex[:8],
        "title": body.title.strip() or "未命名小说",
        "created_at": datetime.datetime.now().isoformat(),
        "idea": body.idea,
        **out,
        "chapters": [],
    }
    for k in ("completed_chapters", "current_chapter_idx", "rewrite_count", "current_draft",
              "review_scores", "review_feedback", "review_pass", "review_issues", "review_suggestions"):
        novel.pop(k, None)
    _save(novel["id"], novel)
    return {"id": novel["id"], "title": novel["title"]}


@app.get("/api/novels/{nid}")
def get_novel(nid: str):
    return _load(nid)


@app.put("/api/novels/{nid}")
def update_novel(nid: str, body: SaveReq):
    novel = _load(nid)
    for k in ("title", "idea", "world_setting", "characters", "outline", "chapters"):
        v = getattr(body, k, None)
        if v is not None:
            novel[k] = v
    _save(nid, novel)
    return {"ok": True}


@app.delete("/api/novels/{nid}")
def delete_novel(nid: str):
    p = NOVELS_DIR / f"{nid}.json"
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.post("/api/novels/{nid}/generate-chapter/{idx}")
async def generate_chapter(nid: str, idx: int, request: Request):
    _apply_settings(request.headers)
    novel = _load(nid)
    if idx < 0 or idx >= len(novel.get("outline", [])):
        raise HTTPException(400, "章节索引超出范围")

    np = _reload()

    state = {
        "user_idea": novel.get("idea", ""),
        "world_setting": novel["world_setting"],
        "characters": novel["characters"],
        "outline": novel["outline"],
        "current_chapter_idx": idx,
        "current_draft": "",
        "completed_chapters": list(novel.get("chapters", [])),
        "review_scores": {"logic_consistency": 0, "pacing": 0, "language_quality": 0, "hook_strength": 0, "character_consistency": 0, "total": 0},
        "review_feedback": "",
        "review_pass": True,
        "review_issues": [],
        "review_suggestions": [],
        "rewrite_count": 0,
    }

    wr = np.write_chapter(state)
    draft = wr["current_draft"]

    rv_state = dict(state, current_draft=draft)
    rv = np.review_chapter(rv_state)

    rewrite_count = 1
    while not rv["review_pass"] and rewrite_count < 3:
        rewrite_count += 1
        wr2 = np.write_chapter(dict(rv_state, rewrite_count=rewrite_count))
        draft = wr2["current_draft"]
        rv = np.review_chapter(dict(rv_state, current_draft=draft, rewrite_count=rewrite_count))

    chapters = list(novel.get("chapters", []))
    while len(chapters) <= idx:
        chapters.append("")
    chapters[idx] = draft

    novel["chapters"] = chapters
    _save(nid, novel)

    return {
        "chapter_index": idx,
        "draft": draft,
        "review": {
            "scores": rv["review_scores"],
            "pass": rv["review_pass"],
            "feedback": rv["review_feedback"],
            "rewrite_count": rewrite_count,
        }
    }
