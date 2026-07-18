# -*- coding: utf-8 -*-
"""
Novel Writing Pipeline -- LangGraph MVP
v2: structured data + refined prompts
"""

import json
import os
import re
import sys
import difflib
import copy
from typing import TypedDict, List, Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# ═══════════════════════════════════════════════
# 0a. env
# ═══════════════════════════════════════════════

_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text("utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip("\"'")
        if _k and not os.environ.get(_k):
            os.environ[_k] = _v

# ═══════════════════════════════════════════════
# 0b. flags + llm
# ═══════════════════════════════════════════════

MOCK = os.getenv("NOVEL_MOCK", "0") == "1" or "--mock" in sys.argv
SHOW_GRAPH = "--show-graph" in sys.argv or "-g" in sys.argv
EXPORT_GRAPH = "--export-graph" in sys.argv

if MOCK:
    llm = None
else:
    _key = os.getenv("OPENAI_API_KEY", "")
    _base = os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", ""))
    if _key and not _key.startswith("sk-your"):
        kwargs = dict(
            model=os.getenv("NOVEL_LLM_MODEL", "gpt-4o-mini"),
            temperature=0.7,
            api_key=_key,
        )
        if _base:
            kwargs["base_url"] = _base
        llm = ChatOpenAI(**kwargs)
    else:
        llm = None
# ═══════════════════════════════════════════════
# 0c. 小说流派定义与流派引导 prompt
# ═══════════════════════════════════════════════

NOVEL_GENRES = {
    "urban": {"label": "都市类（现代背景）", "description": "现代都市为舞台，涵盖异能、系统、重生、职场、校园等子类。", "guide": "## 流派指引：都市类（现代背景）\n背景设定在现代都市，核心张力来自「平凡生活 vs 超常因素」。\n· 世界观需有清晰的现代科技水平与超自然规则的界限\n· 主角应有合理的现代人思维和行为逻辑\n· 冲突可来自：系统任务、异能觉醒、商业竞争、校园人际\n· 风格以轻松幽默或热血爽快为主"},
    "wuxia": {"label": "武侠类（古代/架空江湖）", "description": "古代或架空江湖世界，以内功、门派、武林争霸为核心。", "guide": "## 流派指引：武侠类（古代/架空江湖）\n背景在江湖世界，以内功修炼、外功招式、各派武学为核心。\n· 世界观需有清晰的江湖势力格局（门派/帮会/世家/魔教）\n· 武力体系应有层次感（不入流→二流→一流→绝顶）\n· 核心主题围绕「侠义精神」：恩怨、忠义、复仇、守护\n· 情节推动依靠：秘籍争夺、门派纷争、武林大会"},
    "xianxia": {"label": "仙侠类（修真/神话体系）", "description": "以修真修仙为核心，境界突破、渡劫飞升、仙魔之争。", "guide": "## 流派指引：仙侠类（修真/神话体系）\n背景在修真世界，以境界修炼、渡劫飞升、法宝神通为核心。\n· 境界体系需清晰：炼气→筑基→金丹→元婴→化神→渡劫\n· 世界观应有：修真门派、散修阵营、仙魔对立、上古遗迹\n· 力量来源：灵气、功法、丹药、法宝、阵法\n· 时间尺度以百年/千年计，应有完整的修真社会生态"},
    "mystery": {"label": "推理/悬疑类（核心是解谜）", "description": "以解谜破案为核心，包含本格推理、悬疑惊悚、都市怪谈等。", "guide": "## 流派指引：推理/悬疑类（核心是解谜）\n核心驱动力是「解谜」——每章围绕一个谜题或案件展开。\n· 每章应有明确的谜题或悬念钩子\n· 线索设置应公平，避免机械降神\n· 角色用推理能力/观察力推动剧情\n· 氛围营造关键：阴森场景、诡异细节、心理描写\n· 结尾反转要有伏笔铺垫"},
    "romance": {"label": "青春恋爱轻小说", "description": "以校园或青春为舞台，聚焦情感发展和人际关系。", "guide": "## 流派指引：青春恋爱轻小说\n以校园/青春为舞台，聚焦人物情感与人际关系的细腻发展。\n· 背景通常在现代校园或青春生活场景，轻松明快\n· 核心是「关系进展」：从相遇→相识→好感→波折→确认\n· 人物个性要鲜明讨喜，互动要有化学反应\n· 冲突来自：误会、情敌、家庭压力、自我成长\n· 风格偏轻快，对话生动，内心戏丰富"},
    "fanfic": {"label": "同人文（进入原作世界）", "description": "主角穿越或进入某个既有小说/动漫世界，在原作世界中展开新的故事。", "guide": "## 流派指引：同人文（进入原作世界）\n这是基于用户提供的原作名称与设定摘要创作的同人起点。\n· 只能基于用户输入的原作名、世界设定摘要和角色关系去重建，不要直接复制受版权保护的原作文本\n· 世界观要保留原作的核心规则、势力结构、时代背景和关键舞台\n· 主角通常是穿越者、转生者或外来者，拥有改变原作走向的机会\n· 叙事重点是：融入原作世界、改变既定剧情、与原作角色互动\n· 需要明确说明主角在原作中的身份、切入时间点、既有剧情是否已发生"},
}

DEFAULT_GENRE = "urban"


# ═══════════════════════════════════════════════
# 1. 结构化数据类型 (JSON Schema)
# ═══════════════════════════════════════════════


class CharacterCard(TypedDict):
    name: str
    role: str              # 主角 / 关键配角 / 反派
    archetype: str         # 性格原型（谨慎型 / 热血型 / 智谋型…）
    appearance: str        # 外貌描述（30字）
    personality: str       # 性格特点（50字）
    background: str        # 背景故事（50字）
    motivation: str        # 核心动机（30字）
    abilities: str         # 能力/金手指（50字）
    flaws: str             # 性格缺陷（30字）
    arc: str               # 成长弧线（50字）


class WorldSetting(TypedDict):
    name: str              # 世界观名称
    genre: str             # 题材
    premise: str           # 一句话核心设定
    rules: str             # 世界规则和限制（150字）
    locations: List[str]   # 关键地点
    factions: List[str]    # 势力
    power_system: str      # 力量体系（150字）
    tone: str              # 风格基调


class ChapterOutline(TypedDict):
    chapter_number: int
    title: str              # 章节标题
    summary: str            # 一句话梗概
    conflict: str           # 核心冲突
    hook: str               # 结尾钩子
    key_scenes: List[str]   # 关键场景
    characters_involved: List[str]  # 登场角色


class ReviewScores(TypedDict): # dims 1-5
    logic_consistency: int  # 1-5
    pacing: int             # 1-5
    language_quality: int   # 1-5
    hook_strength: int      # 1-5
    total: int              # 1-10


class ReviewResult(TypedDict):
    scores: ReviewScores
    feedback: str
    pass_: bool
    issues: List[str]
    rewrite_suggestions: List[str]



class VolumeOutline(TypedDict):
    name: str
    arc: str
    chapter_start: int
    chapter_end: int
    core_conflict: str
    climax: str
    key_events: list

class PacingBeat(TypedDict):
    name: str
    chapter_start: int
    chapter_end: int
    purpose: str
    major_turning_point: str
    tension_goal: str

class StoryBible(TypedDict):
    timeline: list
    character_states: list
    open_threads: list
    resolved_threads: list
    locations: list
    continuity_notes: str
    last_chapter_summary: str

class OutlineBundle(TypedDict):
    world_setting: WorldSetting
    characters: list
    volume_outline: VolumeOutline
    pacing_outline: list
    story_bible: StoryBible
    outline: list

class NovelState(TypedDict):
    # 输入
    user_idea: str
    genre: str

    # 角色状态
    character_states: list = []

    # 大纲阶段
    world_setting: WorldSetting
    characters: List[CharacterCard]
    volume_outline: VolumeOutline
    pacing_outline: List[PacingBeat]
    story_bible: StoryBible
    outline: List[ChapterOutline]

    # 写作阶段
    current_chapter_idx: int
    current_draft: str
    completed_chapters: List[str]

    # 审稿阶段
    review_scores: ReviewScores
    review_feedback: str
    review_pass: bool
    review_issues: List[str]
    review_suggestions: List[str]
    continuity_pass: bool
    continuity_issues: List[str]
    continuity_suggestions: List[str]
    rewrite_count: int


# ═══════════════════════════════════════════════
# 2. 节点函数
# ═══════════════════════════════════════════════


def generate_outline(state: NovelState) -> dict:
    """Agent 角色：创意架构师 → 输出结构化世界观 + 角色卡 + 篇章/章节大纲"""
    if MOCK:
        return _mock_outline()

    genre_key = state.get("genre", "")
    genre_guide = NOVEL_GENRES.get(genre_key, NOVEL_GENRES[DEFAULT_GENRE])["guide"]
    extra_context = ""
    if genre_key == "fanfic":
        source_title = (state.get("source_title") or "").strip()
        source_summary = (state.get("source_summary") or "").strip()
        extra_context = (
            "## 同人设定补充\n"
            f"- 原作名称：{source_title or '未填写'}\n"
            f"- 用户提供的原作设定摘要：{source_summary or '未填写'}\n"
            "- 只允许基于以上摘要进行重建，不要照搬原作全文或逐字复制设定。\n"
            "- 重点写出主角进入原作世界后的切入点、身份、与原作角色的互动关系。\n\n"
        )
    prompt = (
        "# Role: 资深网文编辑 / 创意架构师\n\n"
        f"{genre_guide}\n"
        f"{extra_context}"
        "你需要根据用户的一句话创意，设计一套完整的小说起始设定。\n"
        "其中必须同时输出：\n"
        "1. 一个覆盖前 30-50 章的篇章大纲（volume_outline）\n"
        "2. 将篇章拆成 4-6 个节奏段落（pacing_outline，每段 5-10 章）\n"
        "3. 基于节奏段落拆分出的前 5 章章节大纲（outline）\n"
        "章节大纲必须严格服务于篇章大纲，不能彼此割裂。\n"
        "输出必须严格遵循以下 JSON Schema，不得包含 schema 之外的字段：\n\n"
        "```json\n"
        '{\n'
        '  "world_setting": {\n'
        '    "name": "世界观名称",\n'
        '    "genre": "题材（都市/修仙/科幻/末世/玄幻/历史等）",\n'
        '    "premise": "一句话核心设定，概括这个世界最独特的规则",\n'
        '    "rules": "世界的基础规则和限制（150字以内）",\n'
        '    "locations": ["关键地点1：简述", "关键地点2：简述"],\n'
        '    "factions": ["势力1：简述其立场和目标", "势力2：简述"],\n'
        '    "power_system": "力量体系或金手指机制说明（150字以内）",\n'
        '    "tone": "风格基调（如：轻松幽默 / 暗黑压抑 / 热血激昂 / 悬疑烧脑）"\n'
        '  },\n'
        '  "characters": [\n'
        '    {\n'
        '      "name": "姓名",\n'
        '      "role": "主角",\n'
        '      "archetype": "性格原型（如：谨慎型 / 热血型 / 智谋型 / 腹黑型）",\n'
        '      "appearance": "外貌描述（30字以内）",\n'
        '      "personality": "性格特点（50字以内）",\n'
        '      "background": "背景故事（50字以内）",\n'
        '      "motivation": "核心动机（30字以内）",\n'
        '      "abilities": "能力或金手指（50字以内）",\n'
        '      "flaws": "性格缺陷（30字以内）",\n'
        '      "arc": "成长弧线：从什么状态成长为什么状态（50字以内）"\n'
        '    }\n'
        "    // 至少 1 个主角，1-2 个关键配角或反派\n"
        "  ],\n"
        '  "volume_outline": {\n'
        '    "name": "第一篇章名称",\n'
        '    "arc": "本篇章 30-50 章的整体推进路线",\n'
        '    "chapter_start": 1,\n'
        '    "chapter_end": 40,\n'
        '    "core_conflict": "本篇章的核心矛盾",\n'
        '    "climax": "本篇章高潮事件",\n'
        '    "key_events": ["关键事件1", "关键事件2"]\n'
        '  },\n'
        '  "pacing_outline": [\n'
        '    {\n'
        '      "name": "开局钩子段",\n'
        '      "chapter_start": 1,\n'
        '      "chapter_end": 5,\n'
        '      "purpose": "这一段在整篇章中的叙事作用",\n'
        '      "major_turning_point": "本段最大转折",\n'
        '      "tension_goal": "本段张力目标"\n'
        '    }\n'
        '  ],\n'
        '  "outline": [\n'
        '    {\n'
        '      "chapter_number": 1,\n'
        '      "title": "章节标题",\n'
        '      "summary": "一句话梗概（30字以内）",\n'
        '      "conflict": "核心冲突（30字以内）",\n'
        '      "hook": "结尾钩子（20字以内）",\n'
        '      "key_scenes": ["关键场景1", "关键场景2"],\n'
        '      "characters_involved": ["角色名1", "角色名2"]\n'
        '    }\n'
        "    // 共 5 章\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        f"## 用户创意\n{state['user_idea']}\n\n"
        "## 质量要求\n"
        "- 世界观要有一个独特的核心规则（hook）\n"
        "- 角色必须有缺点和成长空间\n"
        "- 篇章大纲必须是 30-50 章范围的一整卷，避免只写 5 章\n"
        "- 节奏段落必须覆盖整篇章，且每段都有清晰叙事作用和转折\n"
        "- 章节大纲只输出前 5 章，且要准确承接第一节奏段的起承转合\n"
        "- 每章必须有一个明确的冲突和一个让人想读下一章的钩子\n"
        "- 5 章构成一个完整的故事弧线（起→承→转→合→高潮）\n"
        "- 只输出 JSON，不要任何额外文字"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    content = _parse_json(response.content)
    bundle = _safe_outline_bundle(content, _mock_outline())
    if not bundle.get("pacing_outline"):
        bundle["pacing_outline"] = _default_pacing_outline(bundle["volume_outline"])
    bundle["story_bible"] = _default_story_bible(bundle["world_setting"], bundle["characters"])
    bundle["outline"] = _normalize_chapter_outlines(bundle.get("outline", []), 1)
    return {
        "world_setting": bundle["world_setting"],
        "characters": bundle["characters"],
        "volume_outline": bundle["volume_outline"],
        "pacing_outline": bundle["pacing_outline"],
        "story_bible": bundle["story_bible"],
        "outline": bundle["outline"],
        "completed_chapters": [],
        "current_chapter_idx": 0,
        "rewrite_count": 0,
    }




def _mock_expand(next_num, count, volume_outline=None):
    result = []
    vol_name = (volume_outline or {}).get("name", "")
    vol_arc = (volume_outline or {}).get("arc", "")
    key_events = (volume_outline or {}).get("key_events", [])
    for i in range(count):
        cn = next_num + i
        if key_events:
            ev = key_events[i % len(key_events)]
            title = f"第{cn}章"
            summary = f"{ev}——{vol_arc[:20]}。"
        else:
            title, summary = f"第{cn}章", f"推进{vol_name}主线。"
        result.append({"chapter_number": cn, "title": title, "summary": summary,
            "conflict": "核心冲突", "hook": "留下悬念",
            "key_scenes": ["开场", "发展", "高潮"],
            "characters_involved": ["主角"]})
    return result


def _mock_assist_update(novel: dict, target: str, instructions: str) -> dict:
    updated = {}
    note = instructions.strip() or "按用户要求优化"
    if target in ("all", "world"):
        ws = dict(novel.get("world_setting", {}))
        ws["premise"] = (ws.get("premise", "") + f"（AI辅助更新：{note}）").strip()
        updated["world_setting"] = ws
    if target in ("volume_new",):
        vols = novel.get("volume_outlines") or ([novel.get("volume_outline")] if novel.get("volume_outline") else [])
        last_end = max([int((v or {}).get("chapter_end", 0) or 0) for v in vols] or [0])
        updated["append_volume_outlines"] = [{
            "name": "新篇章",
            "arc": note,
            "chapter_start": last_end + 1,
            "chapter_end": last_end + 30,
            "core_conflict": "待完善",
            "climax": "待完善",
            "key_events": ["待完善"]
        }]
    if target in ("volume_delete",):
        vols = list(novel.get("volume_outlines") or ([novel.get("volume_outline")] if novel.get("volume_outline") else []))
        updated["volume_outlines"] = vols[:-1] if len(vols) > 1 else []
    if target in ("characters_new",):
        updated["append_characters"] = [{
            "name": "新角色",
            "role": "关键配角",
            "archetype": "待完善",
            "appearance": "根据修改方向新建的角色",
            "personality": note,
            "background": "与主线存在新的剧情关联。",
            "motivation": "推动新冲突。",
            "abilities": "待设定",
            "flaws": "待完善",
            "arc": "从登场到影响主线。"
        }]
    if target in ("all", "characters"):
        chars = [dict(c) for c in novel.get("characters", [])]
        if "新建" in note or "新增" in note or "添加" in note:
            updated["append_characters"] = [{
                "name": "新角色",
                "role": "关键配角",
                "archetype": "待完善",
                "appearance": "根据修改方向新建的角色",
                "personality": note,
                "background": "与主线存在新的剧情关联。",
                "motivation": "推动新冲突。",
                "abilities": "待设定",
                "flaws": "待完善",
                "arc": "从登场到影响主线。"
            }]
        elif chars:
            chars[0]["arc"] = (chars[0].get("arc", "") + f"（调整方向：{note}）").strip()
            updated["characters"] = chars
    if target in ("outline_new",):
        vol = dict(novel.get("volume_outline", {}))
        existing = novel.get("outline", []) or []
        next_num = (existing[-1].get("chapter_number", len(existing)) + 1) if existing else 1
        updated["append_outline"] = _mock_expand(next_num, 3, vol)
    if target in ("all", "outline"):
        vol = dict(novel.get("volume_outline", {}))
        vol["arc"] = (vol.get("arc", "") + f"（调整方向：{note}）").strip()
        updated["volume_outline"] = vol
        updated["pacing_outline"] = novel.get("pacing_outline", []) or _default_pacing_outline(vol)
        if "新建" in note or "新增" in note or "添加" in note:
            existing = novel.get("outline", []) or []
            next_num = (existing[-1].get("chapter_number", len(existing)) + 1) if existing else 1
            updated["append_outline"] = _mock_expand(next_num, 3, vol)
        else:
            updated["outline"] = novel.get("outline", [])
    if target in ("all", "bible"):
        bible = dict(novel.get("story_bible", {}))
        bible["continuity_notes"] = (bible.get("continuity_notes", "") + f"\nAI辅助更新要求：{note}").strip()
        updated["story_bible"] = bible
    return updated


def _compact_text(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _safe_story_bible(raw, fallback_bible: dict, chapter: dict | None = None, draft: str = "", blacklist_entry: dict | None = None) -> dict:
    bible = copy.deepcopy(fallback_bible or {})
    if isinstance(raw, dict):
        for key in ("timeline", "character_states", "open_threads", "resolved_threads", "locations"):
            if isinstance(raw.get(key), list):
                bible[key] = raw[key]
        if isinstance(raw.get("continuity_notes"), str):
            bible["continuity_notes"] = raw["continuity_notes"]
        if isinstance(raw.get("last_chapter_summary"), str):
            bible["last_chapter_summary"] = raw["last_chapter_summary"]
    if chapter:
        chapter_no = chapter.get("chapter_number", "")
        title = chapter.get("title", "")
        summary = chapter.get("summary") or re.sub(r"\s+", " ", draft or "").strip()[:120]
        entry = f"第{chapter_no}章 {title}：{summary}".strip()
        bible["timeline"] = list(dict.fromkeys((bible.get("timeline", []) or []) + [entry]))[-200:]
        bible["last_chapter_summary"] = entry
        hook = (chapter.get("hook") or "").strip()
        if hook:
            bible["open_threads"] = list(dict.fromkeys((bible.get("open_threads", []) or []) + [hook]))[-50:]
    if blacklist_entry:
        black = list(bible.get("blacklist", []) or [])
        black.append(blacklist_entry)
        bible["blacklist"] = black[-200:]
    return bible


def _safe_json_dict(raw, fallback: dict, allowed_keys: set[str] | None = None) -> dict:
    if not isinstance(raw, dict):
        return copy.deepcopy(fallback or {})
    data = copy.deepcopy(fallback or {})
    for key, value in raw.items():
        if allowed_keys and key not in allowed_keys:
            continue
        data[key] = value
    return data


def _safe_outline_bundle(raw: dict, fallback: dict) -> dict:
    bundle = {
        "world_setting": copy.deepcopy(fallback.get("world_setting", {})),
        "characters": copy.deepcopy(fallback.get("characters", [])),
        "volume_outline": copy.deepcopy(fallback.get("volume_outline", {})),
        "pacing_outline": copy.deepcopy(fallback.get("pacing_outline", [])),
        "story_bible": copy.deepcopy(fallback.get("story_bible", {})),
        "outline": copy.deepcopy(fallback.get("outline", [])),
    }
    if not isinstance(raw, dict):
        return bundle
    if isinstance(raw.get("world_setting"), dict):
        bundle["world_setting"] = raw["world_setting"]
    if isinstance(raw.get("characters"), list):
        bundle["characters"] = raw["characters"]
    if isinstance(raw.get("volume_outline"), dict):
        bundle["volume_outline"] = raw["volume_outline"]
    if isinstance(raw.get("pacing_outline"), list):
        bundle["pacing_outline"] = raw["pacing_outline"]
    if isinstance(raw.get("story_bible"), dict):
        bundle["story_bible"] = raw["story_bible"]
    if isinstance(raw.get("outline"), list):
        bundle["outline"] = raw["outline"]
    return bundle


def _chapter_event_record(novel: dict, idx: int, draft: str, result: dict | None = None) -> dict:
    outline = list(novel.get("outline", []) or [])
    chapter = outline[idx] if idx < len(outline) and isinstance(outline[idx], dict) else {}
    chars = chapter.get("characters_involved", []) or []
    scenes = chapter.get("key_scenes", []) or []
    event = {
        "chapter_number": chapter.get("chapter_number", idx + 1),
        "title": chapter.get("title", f"第{idx + 1}章"),
        "summary": chapter.get("summary", ""),
        "conflict": chapter.get("conflict", ""),
        "hook": chapter.get("hook", ""),
        "characters_involved": chars,
        "key_scenes": scenes,
        "draft_excerpt": re.sub(r"\s+", " ", draft or "").strip()[:280],
    }
    if result:
        review = result.get("review_scores", {}) or {}
        event["review_total"] = review.get("total", result.get("review_score", 0))
        event["continuity_pass"] = result.get("continuity_pass", True)
    return event


def _blacklist_tags(event: dict, source: str = "chapter") -> list[str]:
    tags = [f"来源:{source}"]
    conflict = str(event.get("conflict", "") or "").strip()
    hook = str(event.get("hook", "") or "").strip()
    for name in (event.get("characters_involved", []) or [])[:4]:
        if name:
            tags.append(f"角色:{name}")
    if conflict:
        tags.append(f"冲突:{conflict[:16]}")
    if hook:
        tags.append(f"钩子:{hook[:16]}")
    return tags


def _event_blacklist_entry(event: dict) -> dict:
    if not isinstance(event, dict):
        return {}
    entry = {
        "source": "chapter",
        "level": "event",
        "chapter_number": event.get("chapter_number"),
        "title": event.get("title", ""),
        "summary": event.get("summary", ""),
        "conflict": event.get("conflict", ""),
        "hook": event.get("hook", ""),
        "characters_involved": event.get("characters_involved", []) or [],
        "key_scenes": event.get("key_scenes", []) or [],
    }
    entry["tags"] = _blacklist_tags(entry, "chapter")
    return entry


def assist_update_novel(novel: dict, target: str, instructions: str) -> dict:
    """Agent 角色：设定协同编辑 → 按用户方向重写指定设定模块。"""
    target = target if target in {"all", "world", "characters", "characters_new", "volume_new", "volume_delete", "outline", "outline_new", "bible"} else "all"
    if MOCK or llm is None or novel.get("__force_mock"):
        return _mock_assist_update(novel, target, instructions)

    target_desc = {
        "all": "全部核心设定：world_setting、characters、volume_outline、pacing_outline、outline、story_bible",
        "world": "只修改 world_setting",
        "characters": "只修改 characters",
        "characters_new": "只新增角色卡：返回 append_characters，不要覆盖原 characters",
        "volume_new": "只新增篇章大纲：返回 append_volume_outlines，不要覆盖原 volume_outlines",
        "volume_delete": "只删除或精简篇章大纲：返回删除后的 volume_outlines，不要修改章节正文",
        "outline": "只修改大纲：volume_outline、pacing_outline、outline",
        "outline_new": "只新增章节大纲或节奏段：返回 append_outline 或 append_pacing_outline，不要覆盖原 outline",
        "bible": "只修改 story_bible",
    }[target]
    prompt = (
        "# Role: 网文设定协同编辑 / 结构化改稿 Agent\n\n"
        "你会根据用户的修改方向，对现有小说设定做定向重写。"
        "不要推翻用户没有要求修改的部分；涉及联动时要保持世界观、角色、篇章节奏、章节大纲和剧情圣经互相兼容。\n\n"
        f"## 修改范围\n{target_desc}\n\n"
        f"## 用户修改方向\n{instructions.strip()}\n\n"
        "## 当前小说数据\n"
        f"{json.dumps({k: novel.get(k) for k in ['idea','world_setting','characters','volume_outline','volume_outlines','pacing_outline','outline','story_bible']}, ensure_ascii=False)}\n\n"
        "## 输出要求\n"
        "- 只输出 JSON，不要解释\n"
        "- 只返回修改范围内的字段\n"
        "- 所有返回字段必须使用原有字段名和结构\n"
        "- 如果用户是“修改角色卡”，默认返回完整 characters 列表，不要只返回一张角色卡\n"
        "- 如果用户要求“新建/新增/添加”角色卡，不要覆盖原 characters，优先返回 append_characters 数组\n"
        "- 如果用户要求“新建/新增/添加”篇章大纲，不要覆盖原 volume_outlines，优先返回 append_volume_outlines 数组\n"
        "- 如果用户要求删除篇章大纲，返回删除后的 volume_outlines 数组，不要删除 outline 或 chapters\n"
        "- 如果用户要求“新建/新增/添加”章节大纲，不要覆盖原 outline，优先返回 append_outline 数组\n"
        "- 如果用户要求新增节奏段，返回 append_pacing_outline 数组\n"
        "- 如果用户明确要求重做/替换，才返回 characters / outline / pacing_outline 等完整替换字段\n"
        "- 大纲章节必须包含 chapter_number/title/summary/conflict/hook/key_scenes/characters_involved\n\n"
        "## 输出 JSON 示例\n"
        "{\n"
        '  "world_setting": {},\n'
        '  "characters": [],\n'
        '  "append_characters": [],\n'
        '  "volume_outline": {},\n'
        '  "volume_outlines": [],\n'
        '  "append_volume_outlines": [],\n'
        '  "pacing_outline": [],\n'
        '  "append_pacing_outline": [],\n'
        '  "outline": [],\n'
        '  "append_outline": [],\n'
        '  "story_bible": {}\n'
        "}\n"
    )
    content = _parse_json(llm.invoke([SystemMessage(content=prompt)]).content)
    if not isinstance(content, dict):
        raise ValueError("AI 返回的更新结果不是 JSON 对象")

    if target == "volume_new":
        append_vols = content.get("append_volume_outlines")
        if isinstance(append_vols, dict):
            content["append_volume_outlines"] = [append_vols]
        elif append_vols is None:
            if isinstance(content.get("volume_outlines"), list):
                existing_count = len(novel.get("volume_outlines") or ([novel.get("volume_outline")] if novel.get("volume_outline") else []))
                candidates = content["volume_outlines"][existing_count:] or content["volume_outlines"]
                content["append_volume_outlines"] = [v for v in candidates if isinstance(v, dict)]
            elif isinstance(content.get("volume_outline"), dict) and content["volume_outline"]:
                content["append_volume_outlines"] = [content["volume_outline"]]

    allowed = {
        "all": {"world_setting", "characters", "append_characters", "volume_outline", "volume_outlines", "append_volume_outlines", "pacing_outline", "append_pacing_outline", "outline", "append_outline", "story_bible"},
        "world": {"world_setting"},
        "characters": {"characters", "append_characters"},
        "characters_new": {"append_characters"},
        "volume_new": {"append_volume_outlines"},
        "volume_delete": {"volume_outlines"},
        "outline": {"volume_outline", "volume_outlines", "append_volume_outlines", "pacing_outline", "append_pacing_outline", "outline", "append_outline"},
        "outline_new": {"append_pacing_outline", "append_outline"},
        "bible": {"story_bible"},
    }[target]
    return {k: v for k, v in content.items() if k in allowed}


def _default_pacing_outline(volume_outline: dict) -> list:
    start = int((volume_outline or {}).get("chapter_start", 1) or 1)
    end = int((volume_outline or {}).get("chapter_end", 40) or 40)
    spans = [
        ("开局钩子段", "建立主角困境、金手指或核心异常", "主角被迫卷入主线", "快速抓住读者"),
        ("规则探索段", "揭示世界规则与代价", "主角第一次主动利用规则", "爽点和危机交替"),
        ("反派压迫段", "让外部压力和隐藏势力登场", "主角遭遇阶段性失败", "持续加压"),
        ("反击升级段", "整合资源并推进人物成长", "主角完成关键选择", "情绪抬升"),
        ("篇章高潮段", "收束伏笔并打出篇章高潮", "主角赢下阶段性胜利并留下新谜题", "强高潮和新钩子"),
    ]
    total = max(end - start + 1, 5)
    step = max(total // len(spans), 1)
    result = []
    cur = start
    for i, (name, purpose, turn, tension) in enumerate(spans):
        seg_end = end if i == len(spans) - 1 else min(end, cur + step - 1)
        result.append({
            "name": name,
            "chapter_start": cur,
            "chapter_end": seg_end,
            "purpose": purpose,
            "major_turning_point": turn,
            "tension_goal": tension,
        })
        cur = seg_end + 1
        if cur > end:
            break
    return result


def sync_pacing_outline_for_outline(state: dict) -> list:
    """Keep pacing beats covering all chapter outlines without rewriting existing beats."""
    outline = state.get("outline", []) or []
    volume = state.get("volume_outline", {}) or {}
    pacing = [dict(b) for b in (state.get("pacing_outline", []) or []) if isinstance(b, dict)]
    if not outline:
        return pacing or _default_pacing_outline(volume)

    chapter_nums = [
        int(ch.get("chapter_number", i + 1) or (i + 1))
        for i, ch in enumerate(outline)
        if isinstance(ch, dict)
    ]
    if not chapter_nums:
        return pacing or _default_pacing_outline(volume)

    first_chapter = min(chapter_nums)
    last_chapter = max(chapter_nums)
    if not pacing:
        volume_for_default = dict(volume)
        volume_for_default["chapter_start"] = int(volume_for_default.get("chapter_start", first_chapter) or first_chapter)
        volume_for_default["chapter_end"] = max(int(volume_for_default.get("chapter_end", last_chapter) or last_chapter), last_chapter)
        return _default_pacing_outline(volume_for_default)

    def beat_end(beat):
        return int(beat.get("chapter_end", 0) or 0)

    pacing.sort(key=lambda b: int(b.get("chapter_start", 0) or 0))
    covered_end = max(beat_end(b) for b in pacing)
    if covered_end >= last_chapter:
        return pacing

    start = covered_end + 1
    templates = [
        ("承压推进段", "承接新增章节，继续抬高外部压力并推动主线", "主角遭遇新的阻碍或代价", "保持危机感和阅读钩子"),
        ("转折升级段", "让新增章节出现阶段性转折，推动角色主动选择", "主角获得新线索或付出代价", "提高爽点密度与冲突强度"),
        ("阶段收束段", "收束新增章节中的小目标，并埋入下一阶段悬念", "阶段冲突得到局部解决", "形成阶段满足感并留下新问题"),
    ]
    while start <= last_chapter:
        end = min(last_chapter, start + 9)
        name, purpose, turn, tension = templates[(len(pacing)) % len(templates)]
        pacing.append({
            "name": name,
            "chapter_start": start,
            "chapter_end": end,
            "purpose": purpose,
            "major_turning_point": turn,
            "tension_goal": tension,
        })
        start = end + 1
    return pacing


def _default_story_bible(world_setting: dict, characters: list) -> dict:
    return {
        "timeline": [],
        "character_states": [
            {
                "name": c.get("name", ""),
                "status": c.get("background", ""),
                "goal": c.get("motivation", ""),
                "relationship_changes": "",
            }
            for c in (characters or [])
            if isinstance(c, dict)
        ],
        "open_threads": [],
        "resolved_threads": [],
        "locations": list((world_setting or {}).get("locations", []) or []),
        "chapter_events": [],
        "blacklist": [],
        "continuity_notes": (world_setting or {}).get("rules", ""),
        "last_chapter_summary": "",
    }


def _format_story_bible(story_bible: dict) -> str:
    bible = story_bible or {}
    lines = []
    for key, label in [
        ("timeline", "时间线"),
        ("character_states", "人物状态"),
        ("open_threads", "未回收伏笔"),
        ("resolved_threads", "已回收伏笔"),
        ("locations", "地点与物件"),
        ("chapter_events", "章节事件账本"),
        ("blacklist", "重复记忆黑名单"),
    ]:
        val = bible.get(key, [])
        if not val:
            continue
        if isinstance(val, list):
            lines.append(f"{label}：")
            for item in val[-8:]:
                if isinstance(item, dict):
                    lines.append("- " + "；".join(f"{k}:{v}" for k, v in item.items() if v))
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(f"{label}：{val}")
    if bible.get("continuity_notes"):
        lines.append(f"连续性备注：{bible.get('continuity_notes')}")
    if bible.get("last_chapter_summary"):
        lines.append(f"上一章摘要：{bible.get('last_chapter_summary')}")
    return "\n".join(lines) if lines else "暂无"


def _recent_chapter_memory(state: dict, limit: int = 5) -> str:
    outline = state.get("outline", []) or []
    completed = state.get("completed_chapters", []) or []
    filled = [(i, body) for i, body in enumerate(completed) if (body or "").strip()]
    if not filled:
        return "暂无"
    rows = []
    for i, body in filled[-limit:]:
        ch = outline[i] if i < len(outline) and isinstance(outline[i], dict) else {}
        compact = re.sub(r"\s+", " ", body).strip()
        excerpt = compact[:180]
        if len(compact) > 300:
            excerpt += " ... " + compact[-120:]
        rows.append(
            f"- 第{i + 1}章 {ch.get('title', '')}："
            f"大纲={ch.get('summary', '')}；冲突={ch.get('conflict', '')}；"
            f"已写片段={excerpt}"
        )
    return "\n".join(rows)


def _upcoming_outline_memory(state: dict, limit: int = 3) -> str:
    outline = state.get("outline", []) or []
    idx = int(state.get("current_chapter_idx", 0) or 0)
    rows = []
    for i in range(idx + 1, min(len(outline), idx + 1 + limit)):
        ch = outline[i] if isinstance(outline[i], dict) else {}
        rows.append(
            f"- 第{i + 1}章 {ch.get('title', '')}："
            f"{ch.get('summary', '')}；钩子={ch.get('hook', '')}"
        )
    return "\n".join(rows) if rows else "暂无"


def _recent_story_events(state: dict, limit: int = 5) -> str:
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    events = bible.get("chapter_events", []) or []
    if not events:
        return "暂无"
    rows = []
    for ev in events[-limit:]:
        if not isinstance(ev, dict):
            continue
        rows.append(
            f"- 第{ev.get('chapter_number', '?')}章 {ev.get('title', '')}："
            f"事件={ev.get('summary', '')}；冲突={ev.get('conflict', '')}；"
            f"角色={', '.join(ev.get('characters_involved', []) or [])}；钩子={ev.get('hook', '')}"
        )
    return "\n".join(rows) if rows else "暂无"


def _chapter_progress_hint(state: dict) -> str:
    outline = state.get("outline", []) or []
    idx = int(state.get("current_chapter_idx", 0) or 0)
    if idx >= len(outline):
        return "暂无"
    ch = outline[idx] if isinstance(outline[idx], dict) else {}
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    recent_events = bible.get("chapter_events", []) or []
    last_event = recent_events[-1] if recent_events and isinstance(recent_events[-1], dict) else {}
    lines = [
        f"当前章目标：第{ch.get('chapter_number', idx + 1)}章 {ch.get('title', '')}",
        f"本章必须新增的信息：{ch.get('summary', '')}",
        f"本章不能重复的最后一条事件：{last_event.get('summary', '')} / {last_event.get('hook', '')}",
    ]
    return "\n".join(lines)


def _similar_story_events(state: dict, limit: int = 5) -> str:
    outline = state.get("outline", []) or []
    idx = int(state.get("current_chapter_idx", 0) or 0)
    if idx >= len(outline):
        return "暂无"
    ch = outline[idx] if isinstance(outline[idx], dict) else {}
    target = " ".join([
        ch.get("title", ""),
        ch.get("summary", ""),
        ch.get("conflict", ""),
        ch.get("hook", ""),
        " ".join(ch.get("key_scenes", []) or []),
        " ".join(ch.get("characters_involved", []) or []),
    ])
    target = re.sub(r"\s+", "", target)
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    scored = []
    for ev in bible.get("chapter_events", []) or []:
        if not isinstance(ev, dict):
            continue
        ev_no = int(ev.get("chapter_number", 0) or 0)
        if ev_no >= idx + 1:
            continue
        event_text = " ".join([
            str(ev.get("title", "")),
            str(ev.get("summary", "")),
            str(ev.get("conflict", "")),
            str(ev.get("hook", "")),
            " ".join(ev.get("key_scenes", []) or []),
            " ".join(ev.get("characters_involved", []) or []),
        ])
        event_text = re.sub(r"\s+", "", event_text)
        if not event_text:
            continue
        score = difflib.SequenceMatcher(None, target[:500], event_text[:500]).ratio()
        if score >= 0.18:
            scored.append((score, ev))
    scored.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for score, ev in scored[:limit]:
        rows.append(
            f"- 相似度{score:.2f}：第{ev.get('chapter_number', '?')}章 {ev.get('title', '')}；"
            f"事件={ev.get('summary', '')}；冲突={ev.get('conflict', '')}；钩子={ev.get('hook', '')}"
        )
    return "\n".join(rows) if rows else "暂无"


def _blacklist_memory(state: dict, limit: int = 8) -> str:
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    black = bible.get("blacklist", []) or []
    if not black:
        return "暂无"
    rows = []
    for ev in black[-limit:]:
        if not isinstance(ev, dict):
            continue
        tags = "、".join(ev.get("tags", []) or [])
        rows.append(
            f"- 第{ev.get('chapter_number', '?')}章 {ev.get('title', '')}："
            f"事件={ev.get('summary', '')}；冲突={ev.get('conflict', '')}；钩子={ev.get('hook', '')}；标签={tags}"
        )
    return "\n".join(rows) if rows else "暂无"


def _chapter_outline_lock(state: dict) -> str:
    outline = state.get("outline", []) or []
    idx = int(state.get("current_chapter_idx", 0) or 0)
    chapter = outline[idx] if idx < len(outline) and isinstance(outline[idx], dict) else {}
    return (
        f"第 {chapter.get('chapter_number', idx + 1)} 章：{chapter.get('title', '')}\n"
        f"梗概：{chapter.get('summary', '')}\n"
        f"核心冲突：{chapter.get('conflict', '')}\n"
        f"关键场景：{'; '.join(chapter.get('key_scenes', []) or [])}\n"
        f"结尾钩子：{chapter.get('hook', '')}\n"
        f"登场角色：{', '.join(chapter.get('characters_involved', []) or [])}"
    )


def _compact_generation_context(state: dict) -> str:
    outline = state.get("outline", []) or []
    idx = int(state.get("current_chapter_idx", 0) or 0)
    chapter = outline[idx] if idx < len(outline) and isinstance(outline[idx], dict) else {}
    beat = _current_pacing_beat(state, chapter.get("chapter_number", idx + 1))
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    events = [x for x in (bible.get("chapter_events", []) or []) if isinstance(x, dict)]
    black = [x for x in (bible.get("blacklist", []) or []) if isinstance(x, dict)]
    completed = [x for x in (state.get("completed_chapters", []) or []) if (x or "").strip()]
    last_tail = re.sub(r"\s+", " ", completed[-1]).strip()[-180:] if completed else "暂无"
    lines = []
    if beat:
        lines.append(f"节奏段：{beat.get('name', '')}；目标：{beat.get('purpose', '')}；张力：{beat.get('tension_goal', '')}")
    lines.append(f"上一章末尾：{last_tail}")
    if bible.get("last_chapter_summary"):
        lines.append(f"上一章摘要：{bible.get('last_chapter_summary')}")
    if events:
        lines.append("最近事件：")
        for ev in events[-3:]:
            lines.append(f"- 第{ev.get('chapter_number', '?')}章：{ev.get('summary', '')}；冲突={ev.get('conflict', '')}；钩子={ev.get('hook', '')}")
    similar = _similar_story_events(state, limit=3)
    if similar != "暂无":
        lines.append("相似旧桥段，必须避开：")
        lines.append(similar)
    if black:
        lines.append("重复黑名单：")
        for ev in black[-5:]:
            lines.append(f"- {ev.get('summary', '')} / {ev.get('conflict', '')} / {ev.get('hook', '')}")
    audit = state.get("audit_report") or {}
    if isinstance(audit, dict) and audit.get("issues"):
        current_no = chapter.get("chapter_number", idx + 1)
        related = []
        for issue in audit.get("issues", []) or []:
            nums = issue.get("chapter_numbers", []) if isinstance(issue, dict) else []
            if current_no in nums:
                related.append(issue)
        if related:
            lines.append("本章审稿修正记忆：")
            for issue in related[:3]:
                lines.append(f"- {issue.get('type', '问题')}：{issue.get('suggestion') or issue.get('title', '')}")
    return "\n".join(lines) if lines else "暂无"


def _compact_validation_context(state: dict) -> str:
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    parts = [_chapter_outline_lock(state)]
    if bible.get("continuity_notes"):
        parts.append(f"连续性备注：{_compact_text(bible.get('continuity_notes'), 300)}")
    if bible.get("last_chapter_summary"):
        parts.append(f"上一章摘要：{_compact_text(bible.get('last_chapter_summary'), 180)}")
    events = [x for x in (bible.get("chapter_events", []) or []) if isinstance(x, dict)]
    if events:
        parts.append("最近事件：" + " / ".join(_compact_text(x.get("summary", ""), 80) for x in events[-3:]))
    return "\n".join(parts)


def _detect_repeated_story(state: dict, window: int = 5) -> tuple[list[str], list[str]]:
    draft = re.sub(r"\s+", "", state.get("current_draft", "") or "")
    if not draft:
        return [], []
    issues = []
    suggestions = []
    recent = (state.get("completed_chapters", []) or [])[-window:]
    current_outline = (state.get("outline", []) or [])
    current_idx = int(state.get("current_chapter_idx", 0) or 0)
    current_title = current_outline[current_idx].get("title", "") if current_idx < len(current_outline) and isinstance(current_outline[current_idx], dict) else ""
    current_sig = draft[:700]
    for offset, body in enumerate(recent, start=1):
        prev = re.sub(r"\s+", "", body or "")
        if not prev:
            continue
        prev_sig = prev[:700]
        ratio = difflib.SequenceMatcher(None, current_sig, prev_sig).ratio()
        if ratio >= 0.48:
            chapter_no = current_idx - len(recent) + offset
            issues.append(f"当前章节与第{chapter_no}章内容重复度过高，可能重复描写同一事件或桥段")
            suggestions.append("压缩重复段落，改写为新的推进、冲突或信息增量，不要重讲前文已发生的事")
    if current_title and current_title in draft[:120]:
        issues.append("当前章节开头与章节标题/梗概过于贴近，可能只是重复复述大纲")
        suggestions.append("直接进入新的场景或动作，不要把章节梗概换一种说法再写一遍")
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    events = [x for x in (bible.get("chapter_events", []) or []) if isinstance(x, dict)]
    current_outline_text = ""
    if current_idx < len(current_outline) and isinstance(current_outline[current_idx], dict):
        ch = current_outline[current_idx]
        current_outline_text = re.sub(r"\s+", "", " ".join([
            ch.get("summary", ""),
            ch.get("conflict", ""),
            ch.get("hook", ""),
            " ".join(ch.get("key_scenes", []) or []),
        ]))
    for ev in events[-30:]:
        ev_no = int(ev.get("chapter_number", 0) or 0)
        if ev_no >= current_idx + 1:
            continue
        event_words = []
        for text in [ev.get("summary", ""), ev.get("conflict", ""), ev.get("hook", ""), " ".join(ev.get("key_scenes", []) or [])]:
            event_words.extend([w for w in re.split(r"[，。；、,.;:!?！？\s]+", str(text or "")) if len(w) >= 2])
        event_words = list(dict.fromkeys(event_words))[:10]
        if not event_words:
            continue
        hits = [w for w in event_words if w in draft]
        outline_hits = [w for w in event_words if w in current_outline_text]
        if len(hits) >= max(3, len(event_words) // 2) and len(outline_hits) < max(2, len(hits) // 2):
            issues.append(f"当前章节疑似复用第{ev_no}章已发生事件，只是换了描写方式")
            suggestions.append("保留必要承接信息，但必须写出新的事件结果、关系变化或代价，不能复刻旧事件结构")
            break
    return list(dict.fromkeys(issues)), list(dict.fromkeys(suggestions))


def _outline_alignment_issues(state: dict) -> tuple[list[str], list[str]]:
    outline = state.get("outline", []) or []
    idx = int(state.get("current_chapter_idx", 0) or 0)
    if idx >= len(outline):
        return [], []
    chapter = outline[idx] if isinstance(outline[idx], dict) else {}
    draft = re.sub(r"\s+", "", state.get("current_draft", "") or "")
    if not draft:
        return [], []

    def split_keywords(text: str) -> list[str]:
        parts = []
        for token in re.split(r"[，。；、,.;:!?！？\s]+", str(text or "")):
            token = token.strip()
            if len(token) >= 2 and token not in parts:
                parts.append(token)
        return parts

    def phrase_hit(phrase: str) -> bool:
        phrase = re.sub(r"\s+", "", str(phrase or "")).strip()
        if not phrase:
            return False
        if phrase in draft:
            return True
        if len(phrase) <= 4:
            return False
        fragments = {phrase[:4], phrase[-4:], phrase[: max(3, len(phrase) // 2)]}
        return any(fragment and fragment in draft for fragment in fragments)

    keywords = []
    for field in (
        chapter.get("title", ""),
        chapter.get("summary", ""),
        chapter.get("conflict", ""),
        chapter.get("hook", ""),
    ):
        keywords.extend(split_keywords(field))
    for item in chapter.get("key_scenes", []) or []:
        keywords.extend(split_keywords(item))
    for item in chapter.get("characters_involved", []) or []:
        keywords.extend(split_keywords(item))
    keywords = list(dict.fromkeys(k for k in keywords if len(k) >= 2))

    issues: list[str] = []
    suggestions: list[str] = []

    if keywords:
        hits = [kw for kw in keywords if phrase_hit(kw)]
        min_hits = 2 if len(keywords) < 6 else min(4, max(2, len(keywords) // 3 + 1))
        if len(hits) < min_hits:
            issues.append(
                f"正文与当前章节大纲对齐度偏低：仅命中 {len(hits)}/{len(keywords)} 个大纲关键词，可能没有真正落实本章梗概"
            )
            suggestions.append(
                "请按当前章节大纲重写本章，必须覆盖梗概、冲突、关键场景和结尾钩子，不要沿用旧稿的推进方向"
            )

    scenes = [str(item).strip() for item in (chapter.get("key_scenes", []) or []) if str(item).strip()]
    if scenes:
        scene_hits = sum(1 for item in scenes if phrase_hit(item))
        if scene_hits < max(1, len(scenes) // 2):
            issues.append(
                f"本章未充分覆盖关键场景：仅命中 {scene_hits}/{len(scenes)} 个关键场景"
            )
            suggestions.append("补齐关键场景顺序，优先写本章必须发生的动作节点，不要跳到别的桥段")

    hook = str(chapter.get("hook", "") or "").strip()
    if hook and not phrase_hit(hook):
        issues.append("本章正文没有有效落实章节钩子，结尾方向可能偏离最新大纲")
        suggestions.append("结尾必须回到当前章节钩子，不要用旧冲突或旧收尾替代")

    return list(dict.fromkeys(issues)), list(dict.fromkeys(suggestions))


def _chapter_audit_brief(novel: dict, max_chars: int = 900) -> list:
    outline = novel.get("outline", []) or []
    chapters = novel.get("chapters", []) or []
    briefs = []
    for i, body in enumerate(chapters):
        if not (body or "").strip():
            continue
        ch = outline[i] if i < len(outline) and isinstance(outline[i], dict) else {}
        text = re.sub(r"\s+", " ", body).strip()
        if len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + " ... " + text[-half:]
        briefs.append({
            "chapter_number": i + 1,
            "title": ch.get("title", f"第{i + 1}章"),
            "outline_summary": ch.get("summary", ""),
            "outline_conflict": ch.get("conflict", ""),
            "outline_hook": ch.get("hook", ""),
            "pacing_beat": _current_pacing_beat(novel, i + 1),
            "text_excerpt": text,
        })
    return briefs


def _current_pacing_beat(state: dict, chapter_number: int) -> dict:
    for beat in state.get("pacing_outline", []) or []:
        if int(beat.get("chapter_start", 0) or 0) <= chapter_number <= int(beat.get("chapter_end", 0) or 0):
            return beat
    return {}


def _heuristic_audit_novel(novel: dict) -> dict:
    outline = novel.get("outline", []) or []
    chapters = novel.get("chapters", []) or []
    issues = []
    chapter_map = []
    seen = {}
    for i, body in enumerate(chapters):
        if not (body or "").strip():
            continue
        chapter_no = i + 1
        ch = outline[i] if i < len(outline) and isinstance(outline[i], dict) else {}
        text = re.sub(r"\s+", "", body)
        sig = text[:220]
        if sig and sig in seen:
            issues.append({
                "severity": "high",
                "type": "重复章节",
                "chapter_numbers": [seen[sig], chapter_no],
                "title": "章节正文疑似重复",
                "evidence": "两章开头内容高度相似",
                "suggestion": "检查是否误复制正文，必要时重新生成后一个章节。",
            })
        elif sig:
            seen[sig] = chapter_no
        if ch.get("summary"):
            keywords = [w for w in re.split(r"[，。；、\s]+", ch.get("summary", "")) if len(w) >= 2][:4]
            missing = [w for w in keywords if w not in body]
            if len(missing) >= max(2, len(keywords) // 2):
                issues.append({
                    "severity": "medium",
                    "type": "大纲偏离",
                    "chapter_numbers": [chapter_no],
                    "title": "正文与章节梗概关联较弱",
                    "evidence": f"章节梗概关键词未明显出现：{'、'.join(missing[:4])}",
                    "suggestion": "复查本章正文是否承接章节大纲，或更新章节大纲以匹配正文。",
                })
        chapter_map.append({
            "chapter_number": chapter_no,
            "title": ch.get("title", f"第{chapter_no}章"),
            "status": "有问题" if any(chapter_no in x.get("chapter_numbers", []) for x in issues) else "通过",
        })
    score = max(40, 100 - len(issues) * 12)
    return {
        "score": score,
        "summary": f"已扫描 {len(chapter_map)} 个已生成章节，发现 {len(issues)} 个潜在问题。",
        "issues": issues,
        "chapter_map": chapter_map,
    }


def audit_novel(novel: dict, focus: str = "all") -> dict:
    """Agent 角色：整本小说审稿编辑 → 扫描章节与设定/大纲/剧情圣经的一致性。"""
    fallback = _heuristic_audit_novel(novel)
    if MOCK or llm is None or novel.get("__force_mock"):
        return fallback

    audit_input = {
        "focus": focus,
        "title": novel.get("title", ""),
        "idea": novel.get("idea", ""),
        "world_setting": novel.get("world_setting", {}),
        "characters": novel.get("characters", []),
        "volume_outline": novel.get("volume_outline", {}),
        "volume_outlines": novel.get("volume_outlines", []),
        "pacing_outline": novel.get("pacing_outline", []),
        "story_bible": novel.get("story_bible", {}),
        "chapters": _chapter_audit_brief(novel),
    }
    prompt = (
        "# Role: 网文整本审稿编辑 / 连续性审计 Agent\n\n"
        "你负责扫描已生成章节与世界观、角色卡、篇章大纲、节奏段落、章节大纲、剧情圣经之间的关联性。"
        "请找出：设定冲突、人设冲突、章节大纲偏离、剧情圣经矛盾、重复章节、重复桥段、节奏错位、伏笔遗漏。\n\n"
        "## 审稿数据\n"
        f"{json.dumps(audit_input, ensure_ascii=False)}\n\n"
        "## 输出要求\n"
        "- 只输出 JSON，不要解释，不要 Markdown\n"
        "- issue 必须指向具体章节号\n"
        "- 没有确定证据的问题不要夸大，可以标为 low\n"
        "- chapter_map 为每个已生成章节给一个简短状态\n\n"
        "{\n"
        '  "score": 86,\n'
        '  "summary": "总体审稿结论",\n'
        '  "issues": [\n'
        '    {"severity":"high|medium|low","type":"设定冲突|人设冲突|大纲偏离|重复章节|重复桥段|节奏错位|伏笔问题","chapter_numbers":[1],"title":"问题标题","evidence":"证据","suggestion":"修改建议"}\n'
        "  ],\n"
        '  "chapter_map": [{"chapter_number":1,"title":"章节名","status":"通过|有问题|需复核"}]\n'
        "}\n"
    )
    try:
        content = _parse_json(llm.invoke([SystemMessage(content=prompt)]).content)
    except Exception:
        return fallback
    if not isinstance(content, dict):
        return fallback
    content["score"] = int(content.get("score", fallback["score"]) or fallback["score"])
    content["summary"] = content.get("summary") or fallback["summary"]
    content["issues"] = content.get("issues") if isinstance(content.get("issues"), list) else fallback["issues"]
    content["chapter_map"] = content.get("chapter_map") if isinstance(content.get("chapter_map"), list) else fallback["chapter_map"]
    return content


def _fallback_outline_update_from_audit(novel: dict, report: dict) -> dict:
    outline = [dict(ch) for ch in (novel.get("outline", []) or []) if isinstance(ch, dict)]
    issue_notes = {}
    for issue in report.get("issues", []) or []:
        note = f"{issue.get('type', '审稿问题')}：{issue.get('suggestion') or issue.get('title', '')}"
        for num in issue.get("chapter_numbers", []) or []:
            issue_notes.setdefault(int(num), []).append(note)
    for ch in outline:
        num = int(ch.get("chapter_number", 0) or 0)
        if num in issue_notes:
            ch["summary"] = (ch.get("summary", "") + "（按审稿调整：" + "；".join(issue_notes[num]) + "）").strip()
            ch["conflict"] = ch.get("conflict", "") or "按审稿报告重新校准本章冲突"
    pacing = sync_pacing_outline_for_outline({
        "volume_outline": novel.get("volume_outline", {}),
        "pacing_outline": novel.get("pacing_outline", []),
        "outline": outline,
    })
    bible = _normalize_story_bible(novel.get("story_bible", {}), novel.get("world_setting", {}), novel.get("characters", []))
    old = (bible.get("continuity_notes") or "").strip()
    bible["continuity_notes"] = f"{old}\n已根据最近审稿报告校准章节大纲，重新生成相关章节时需优先避开审稿问题。".strip()
    return {"outline": outline, "pacing_outline": pacing, "story_bible": bible}


def update_outline_from_audit(novel: dict) -> dict:
    """Agent 角色：大纲修订编辑 → 根据审稿报告修正大纲体系，不改正文。"""
    report = novel.get("audit_report") or {}
    if not report or not report.get("issues"):
        return {}

    if MOCK or llm is None or novel.get("__force_mock"):
        return _fallback_outline_update_from_audit(novel, report)

    payload = {
        "world_setting": novel.get("world_setting", {}),
        "characters": novel.get("characters", []),
        "volume_outline": novel.get("volume_outline", {}),
        "volume_outlines": novel.get("volume_outlines", []),
        "pacing_outline": novel.get("pacing_outline", []),
        "outline": novel.get("outline", []),
        "story_bible": novel.get("story_bible", {}),
        "audit_report": report,
    }
    prompt = (
        "# Role: 网文大纲修订编辑\n\n"
        "你要根据整本审稿报告，修订小说的大纲体系。只修改大纲和剧情圣经备注，不要改正文，不要删除章节。\n"
        "重点解决：设定冲突、人设冲突、大纲偏离、重复章节、重复桥段、节奏错位、伏笔问题。\n\n"
        "## 当前数据\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "## 输出要求\n"
        "- 只输出 JSON\n"
        "- 保持章节编号连续，尽量保留原章节数量\n"
        "- 对审稿指出的章节，在 summary/conflict/hook/key_scenes 中体现修正方向\n"
        "- pacing_outline 必须覆盖 outline 的章节范围\n"
        "- story_bible.continuity_notes 要写入后续重新生成章节时必须遵守的审稿修正记忆\n\n"
        "{\n"
        '  "volume_outline": {},\n'
        '  "volume_outlines": [],\n'
        '  "pacing_outline": [],\n'
        '  "outline": [],\n'
        '  "story_bible": {}\n'
        "}\n"
    )
    try:
        content = _parse_json(llm.invoke([SystemMessage(content=prompt)]).content)
    except Exception:
        return _fallback_outline_update_from_audit(novel, report)
    if not isinstance(content, dict):
        return _fallback_outline_update_from_audit(novel, report)
    result = {}
    for key in ("volume_outline", "volume_outlines", "pacing_outline", "outline", "story_bible"):
        if key in content and content[key]:
            result[key] = content[key]
    if not result:
        return _fallback_outline_update_from_audit(novel, report)
    if result.get("outline"):
        result["outline"] = _normalize_chapter_outlines(result["outline"], 1)
        result["pacing_outline"] = sync_pacing_outline_for_outline({
            "volume_outline": result.get("volume_outline", novel.get("volume_outline", {})),
            "pacing_outline": result.get("pacing_outline", novel.get("pacing_outline", [])),
            "outline": result["outline"],
        })
    return result


def _normalize_story_bible(story_bible: dict, world_setting: dict, characters: list) -> dict:
    default = _default_story_bible(world_setting, characters)
    if not isinstance(story_bible, dict):
        return default
    for key, value in default.items():
        story_bible.setdefault(key, value)
    return story_bible


def _build_state_summary(state: dict) -> str:
    vol = state.get("volume_outline", {}) or {}
    bible = _normalize_story_bible(state.get("story_bible", {}), state.get("world_setting", {}), state.get("characters", []))
    outline = state.get("outline", []) or []
    idx = state.get("current_chapter_idx", 0)
    completed = state.get("completed_chapters", []) or []
    parts = []
    if vol:
        parts.append(
            f"篇章：{vol.get('name', '')}（第{vol.get('chapter_start', '?')}-{vol.get('chapter_end', '?')}章）"
        )
        if vol.get("arc"):
            parts.append(f"篇章走向：{vol.get('arc')}")
        if vol.get("core_conflict"):
            parts.append(f"核心矛盾：{vol.get('core_conflict')}")
    if outline and 0 <= idx < len(outline):
        ch = outline[idx]
        beat = _current_pacing_beat(state, ch.get("chapter_number", idx + 1))
        if beat:
            parts.append(
                f"节奏段：{beat.get('name', '')}（第{beat.get('chapter_start', '?')}-{beat.get('chapter_end', '?')}章）"
            )
            parts.append(f"段落目标：{beat.get('purpose', '')} / {beat.get('tension_goal', '')}")
        parts.append(
            f"当前章节：第{ch.get('chapter_number', '?')}章 {ch.get('title', '')}"
        )
        if ch.get("summary"):
            parts.append(f"章节梗概：{ch.get('summary')}")
        if ch.get("conflict"):
            parts.append(f"章节冲突：{ch.get('conflict')}")
        if ch.get("hook"):
            parts.append(f"章节钩子：{ch.get('hook')}")
    if completed:
        parts.append(f"已完成章节数：{len(completed)}")
        parts.append(f"上一章末尾：{completed[-1][:180]}")
    audit = state.get("audit_report") or {}
    if isinstance(audit, dict) and audit.get("issues"):
        current_no = None
        if outline and 0 <= idx < len(outline):
            current_no = outline[idx].get("chapter_number", idx + 1)
        related = []
        for issue in audit.get("issues", []) or []:
            nums = issue.get("chapter_numbers", []) if isinstance(issue, dict) else []
            if current_no in nums or len(related) < 5:
                related.append(issue)
        if related:
            parts.append("## 最近审稿记忆")
            parts.append(f"审稿结论：{audit.get('summary', '')}")
            for issue in related[:6]:
                nums = "、".join(f"第{n}章" for n in issue.get("chapter_numbers", []) or [])
                parts.append(
                    f"- {issue.get('severity', 'medium')} / {issue.get('type', '问题')} / {nums}："
                    f"{issue.get('title', '')}。证据：{issue.get('evidence', '')}。建议：{issue.get('suggestion', '')}"
                )
    parts.append("## 剧情圣经")
    parts.append(_format_story_bible(bible))
    return "\n".join(parts) if parts else "暂无"


def _normalize_chapter_outlines(items, start_num: int) -> list:
    normalized = []
    for i, ch in enumerate(items or []):
        if not isinstance(ch, dict):
            continue
        normalized.append({
            "chapter_number": start_num + i,
            "title": ch.get("title", f"第{start_num + i}章"),
            "summary": ch.get("summary", ""),
            "conflict": ch.get("conflict", ""),
            "hook": ch.get("hook", ""),
            "key_scenes": ch.get("key_scenes", []) or [],
            "characters_involved": ch.get("characters_involved", []) or [],
        })
    return normalized


def _chapter_import_brief(chapters: list, limit: int = 70) -> str:
    lines = []
    total = len(chapters or [])
    for i, ch in enumerate((chapters or [])[:limit]):
        title = ch.get("title", f"第{i + 1}章")
        body = ch.get("body", "") or ""
        head = body[:500]
        mid_start = max((len(body) // 2) - 250, 0)
        mid = body[mid_start:mid_start + 500]
        tail = body[-500:] if len(body) > 500 else body
        lines.append(
            f"## 第{i + 1}章 {title}\n"
            f"字数：{len(body)}\n"
            f"开头：{head}\n"
            f"中段：{mid}\n"
            f"结尾：{tail}\n"
        )
    if total > limit:
        lines.append(f"## 其余章节\n共 {total - limit} 章未展开，请根据已给样本和标题延续归纳，不要编造具体未读细节。")
    return "\n".join(lines)


def _fallback_import_reconstruction(novel: dict, chapters_raw: list) -> dict:
    outline = []
    for i, ch in enumerate(chapters_raw or []):
        body = ch.get("body", "") or ""
        compact = " ".join(body.split())
        outline.append({
            "chapter_number": i + 1,
            "title": ch.get("title") or f"第{i + 1}章",
            "summary": compact[:160],
            "conflict": "根据导入正文自动提取，建议继续使用 AI 辅助更新细化",
            "hook": compact[-100:] if compact else "",
            "key_scenes": [],
            "characters_involved": [],
        })
    volume = dict(novel.get("volume_outline", {}) or {})
    volume.update({
        "name": volume.get("name") or "导入篇章",
        "arc": volume.get("arc") or "根据 TXT 正文导入生成，待进一步精修。",
        "chapter_start": 1,
        "chapter_end": max(len(outline), 1),
        "key_events": [x["summary"] for x in outline[:8] if x.get("summary")],
    })
    return {
        "world_setting": novel.get("world_setting", {}),
        "characters": novel.get("characters", []),
        "volume_outline": volume,
        "volume_outlines": [volume],
        "pacing_outline": novel.get("pacing_outline", []),
        "story_bible": novel.get("story_bible", {}),
        "outline": outline,
    }


def reconstruct_imported_novel(novel: dict, chapters_raw: list) -> dict:
    """Agent 角色：导入档案员 → 从已有正文反向重建小说设定档案。"""
    if MOCK or llm is None:
        return _fallback_import_reconstruction(novel, chapters_raw)

    chapter_brief = _chapter_import_brief(chapters_raw)
    prompt = (
        "# Role: 网文导入档案员 / 反向创作工程师\n"
        "你要根据一部已写好的小说 TXT 正文，反向重建它在创作系统里的结构化档案。"
        "这不是续写，也不是改写正文；正文已经存在，必须以原文事实为准。\n\n"
        "## 任务目标\n"
        "请像重新创建一篇新小说一样，补全：世界观、角色卡、篇章大纲、节奏段落、剧情圣经、章节大纲。"
        "章节大纲必须覆盖所有已导入章节，并和原章节正文对应。\n\n"
        "## 小说基础信息\n"
        f"{json.dumps({k: novel.get(k) for k in ['title', 'idea', 'genre']}, ensure_ascii=False)}\n\n"
        "## 章节正文样本\n"
        f"{chapter_brief}\n\n"
        "## 输出 JSON\n"
        "只输出一个 JSON 对象，不要 Markdown，不要解释。字段结构如下：\n"
        "{\n"
        '  "world_setting": {\n'
        '    "name": "世界/故事舞台名称",\n'
        '    "genre": "类型",\n'
        '    "premise": "故事核心设定",\n'
        '    "rules": "世界规则、限制和代价",\n'
        '    "locations": ["重要地点"],\n'
        '    "factions": ["势力/组织/家族"],\n'
        '    "power_system": "力量/职业/社会机制",\n'
        '    "tone": "文风和情绪基调"\n'
        "  },\n"
        '  "characters": [\n'
        '    {"name":"角色名","role":"定位","archetype":"角色原型","appearance":"外貌","personality":"性格","background":"背景","motivation":"动机","abilities":"能力/资源","flaws":"缺点","arc":"成长线"}\n'
        "  ],\n"
        '  "volume_outline": {"name":"篇章名","arc":"整体剧情走向","chapter_start":1,"chapter_end":章节总数,"core_conflict":"核心矛盾","climax":"高潮事件","key_events":["关键事件"]},\n'
        '  "volume_outlines": [{"name":"篇章名","arc":"整体剧情走向","chapter_start":1,"chapter_end":章节总数,"core_conflict":"核心矛盾","climax":"高潮事件","key_events":["关键事件"]}],\n'
        '  "pacing_outline": [{"name":"节奏段名称","chapter_start":1,"chapter_end":5,"purpose":"叙事作用","major_turning_point":"关键转折","tension_goal":"张力目标"}],\n'
        '  "story_bible": {"timeline":["重要事件时间线"],"character_states":[{"name":"角色名","status":"当前状态","goal":"目标","relationship_changes":"关系变化"}],"open_threads":["未回收伏笔"],"resolved_threads":["已回收伏笔"],"locations":["地点或物件状态"],"continuity_notes":"连续性备注","last_chapter_summary":"最后一章摘要"},\n'
        '  "outline": [{"chapter_number":1,"title":"章节标题","summary":"本章剧情摘要","conflict":"本章冲突","hook":"本章钩子/结尾悬念","key_scenes":["关键场景"],"characters_involved":["登场角色"]}]\n'
        "}\n\n"
        "## 质量要求\n"
        "- 只基于正文样本能确定的事实归纳，不要发明过多新设定。\n"
        "- 角色卡至少包含主角和反复出现的重要角色；如果样本无法确定外貌，可写“正文未明确”。\n"
        "- 章节大纲数量必须等于导入章节数；章节编号从 1 连续递增。\n"
        "- 剧情圣经要记录已有事实、人物状态、伏笔和地点状态，便于后续继续写。\n"
        "- 如果导入章节很多，仍要为每章输出一条章节大纲，可根据标题和样本提炼简洁摘要。\n"
    )
    content = _parse_json(llm.invoke([SystemMessage(content=prompt)]).content)
    fallback = _fallback_import_reconstruction(novel, chapters_raw)
    bundle = _safe_outline_bundle(content, fallback)
    if len(bundle.get("outline", [])) != len(chapters_raw or []):
        bundle["outline"] = fallback["outline"]
    bundle["outline"] = _normalize_chapter_outlines(bundle.get("outline", []), 1)
    bundle["volume_outlines"] = [bundle["volume_outline"]]
    result = bundle
    return result


def expand_chapter_outlines(state, count=5):
    existing = state.get("outline", []) or []
    vol = state.get("volume_outline", {}) or {}
    next_num = (existing[-1]["chapter_number"] + 1) if existing else max(int(vol.get("chapter_start", 1) or 1), 1)
    beat = _current_pacing_beat(state, next_num)

    if globals().get("MOCK", False) or llm is None:
        return _mock_expand(next_num, count, vol)

    recent = existing[-3:]
    recent_text = "\n".join(
        f"- 第{ch.get('chapter_number', '?')}章 {ch.get('title', '')}：{ch.get('summary', '')} / {ch.get('hook', '')}"
        for ch in recent
    ) or "无"
    prompt = (
        "# Role: 资深网文分章编辑\n\n"
        "你负责把篇章大纲拆成连续的章节大纲。\n"
        "要求：\n"
        f"- 只生成 {count} 章\n"
        f"- 起始章节号必须是 {next_num}\n"
        "- 每一章都要严格承接前文节奏，不能跳卷、不能重复前文事件\n"
        "- 必须围绕篇章大纲推进，不要脱离主线\n"
        "- 只输出 JSON，格式必须是 {\"outline\": [...]} \n\n"
        "## 篇章大纲\n"
        f"名称：{vol.get('name', '')}\n"
        f"整体走向：{vol.get('arc', '')}\n"
        f"章节范围：第{vol.get('chapter_start', '?')}-{vol.get('chapter_end', '?')}章\n"
        f"核心矛盾：{vol.get('core_conflict', '')}\n"
        f"高潮：{vol.get('climax', '')}\n"
        f"关键事件：{'; '.join(vol.get('key_events', []) or [])}\n\n"
        "## 当前节奏段\n"
        f"名称：{beat.get('name', '')}\n"
        f"范围：第{beat.get('chapter_start', '?')}-{beat.get('chapter_end', '?')}章\n"
        f"叙事作用：{beat.get('purpose', '')}\n"
        f"转折：{beat.get('major_turning_point', '')}\n"
        f"张力目标：{beat.get('tension_goal', '')}\n\n"
        "## 剧情圣经\n"
        f"{_format_story_bible(state.get('story_bible', {}))}\n\n"
        "## 最近章节\n"
        f"{recent_text}\n\n"
        "## 角色\n"
        + "\n".join(
            f"- {c.get('name', '')}（{c.get('role', '')}）：{c.get('personality', '')}。能力：{c.get('abilities', '')}"
            for c in (state.get("characters", []) or [])
        )
        + "\n\n"
        "输出示例：\n"
        "{\n"
        '  "outline": [\n'
        "    {\n"
        '      "chapter_number": 6,\n'
        '      "title": "章节标题",\n'
        '      "summary": "一句话梗概",\n'
        '      "conflict": "核心冲突",\n'
        '      "hook": "结尾钩子",\n'
        '      "key_scenes": ["场景1", "场景2"],\n'
        '      "characters_involved": ["角色名"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    content = _parse_json(response.content)
    items = content.get("outline", content if isinstance(content, list) else [])
    return _normalize_chapter_outlines(items, next_num)

def write_chapter(state: NovelState) -> dict:
    """Agent 角色：畅销网文作家 → 写出吸引人的章节"""
    idx = state["current_chapter_idx"]
    chapter = state["outline"][idx]

    if MOCK:
        return _mock_chapter(idx, chapter)

    compact_context = _compact_generation_context(state)

    # 格式化角色卡供 prompt 使用
    involved = set(chapter.get("characters_involved", []) or [])
    selected_chars = [
        c for c in state.get("characters", [])
        if not involved or c.get("name") in involved or c.get("role") == "主角"
    ][:6]
    chars_text = "\n".join(
        f"- {c.get('name', '')}（{c.get('role', '')}）: {c.get('personality', '')}。能力：{c.get('abilities', '')}。"
        for c in selected_chars
    )

    prompt = (
        "# Role: 畅销网文作家\n"
        "写长篇网文正文，重点是推进新事件、避免重复旧桥段。\n\n"
        "## 世界观\n"
        f"名称：{state['world_setting'].get('name', '')}\n"
        f"风格：{state['world_setting'].get('tone', '')}\n"
        f"核心规则：{_compact_text(state['world_setting'].get('rules', ''), 220)}\n"
        f"力量体系：{_compact_text(state['world_setting'].get('power_system', ''), 220)}\n\n"
        "## 角色\n"
        f"{chars_text or '暂无'}\n\n"
        "## 当前章节\n"
        f"{_chapter_outline_lock(state)}\n\n"
        "## 必要记忆\n"
        f"{compact_context}\n\n"
        "## 写作规范\n"
        "1. 正文 3000 字左右，节奏快慢交替\n"
        "2. 用对话和行动推动剧情，少用解释性旁白\n"
        "3. 结尾必须自然引出指定的钩子\n"
        "4. 必须落实本章梗概、冲突、关键场景，不要偏离章节大纲\n"
        "5. 禁止把最近事件、相似旧桥段或黑名单内容换一种说法再写一遍\n"
        "6. 如需承接旧事件，只能用几句话带过，正文必须产生新的信息、关系变化或代价\n"
        "7. 禁止出现“总结一下”“接下来”“如前所述”等元叙述词\n"
        "8. 直接输出小说正文，不要任何标题或说明文字"
    )
    draft = llm.invoke([SystemMessage(content=prompt)]).content
    return {"current_draft": draft}


def review_chapter(state: NovelState) -> dict:
    """Agent 角色：审稿编辑 → 结构化评分 + 具体修改建议"""
    if MOCK:
        return _mock_review()

    if os.getenv("NOVEL_FAST_REVIEW", "1") == "1":
        rep_issues, rep_suggestions = _detect_repeated_story(state)
        outline_issues, outline_suggestions = _outline_alignment_issues(state)
        draft = state.get("current_draft", "") or ""
        issues = list(dict.fromkeys(rep_issues + outline_issues))
        suggestions = list(dict.fromkeys(rep_suggestions + outline_suggestions))
        if len(draft.strip()) < 600:
            issues.append("正文篇幅过短，可能没有完整展开本章冲突")
            suggestions.append("扩写本章关键场景和人物互动，保证章节有完整起伏")
        if any(x in draft for x in ("总结一下", "接下来", "如前所述")):
            issues.append("正文出现元叙述词，阅读体验像说明文")
            suggestions.append("删除元叙述，改为角色行动、对话或场景推进")
        passed = not issues
        total = 8 if passed else 6
        return {
            "review_scores": {
                "logic_consistency": 4 if passed else 3,
                "pacing": 4 if passed else 3,
                "character_consistency": 4 if passed else 3,
                "language_quality": 4 if passed else 3,
                "hook_strength": 4 if passed else 3,
                "total": total,
            },
            "review_score": total,
            "review_feedback": "快速验收通过" if passed else "快速验收发现需要修复的问题",
            "review_pass": passed,
            "review_issues": issues,
            "review_suggestions": suggestions,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    prompt = (
        "# Role: 资深网文审稿编辑\n"
        "检查章节是否可发布，重点看：是否贴合大纲、是否重复旧剧情、是否有明显逻辑问题。\n\n"
        "## 输出 JSON\n"
        "{\n"
        '  "scores": {\n'
        '    "logic_consistency": 4,\n'
        '    "pacing": 3,\n'
        '    "language_quality": 4,\n'
        '    "hook_strength": 3\n'
        "  },\n"
        '  "total": 7,\n'
        '  "feedback": "综合评语（50字以内）",\n'
        '  "pass": false,\n'
        '  "issues": ["具体问题1", "具体问题2"],\n'
        '  "rewrite_suggestions": ["改进建议1", "改进建议2"]\n'
        "}\n"
        "判定标准：偏离大纲、重复前文事件、换皮复用旧桥段、钩子未落实，均 pass=false。\n\n"
        "## 待审章节\n"
        f"{state['current_draft']}\n\n"
        "## 规划与记忆约束\n"
        f"{_compact_validation_context(state)}\n\n"
        "直接输出 JSON，不要任何额外文字。"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    try:
        content = _parse_json(response.content)
    except ValueError:
        raw = _compact_text(response.content)
        return {
            "review_scores": {
                "logic_consistency": 3,
                "pacing": 3,
                "character_consistency": 3,
                "language_quality": 3,
                "hook_strength": 3,
                "total": 7,
            },
            "review_score": 7,
            "review_feedback": f"审稿返回未按JSON格式输出，已保留原始意见：{raw}",
            "review_pass": True,
            "review_issues": [],
            "review_suggestions": [],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    sc = content["scores"]
    total_score = content.get("total", sum(sc.values()) // 2)
    return {
        "review_scores": {
            "logic_consistency": sc.get("logic_consistency", 3),
            "pacing": sc.get("pacing", 3),
            "character_consistency": sc.get("character_consistency", 3),
            "language_quality": sc.get("language_quality", 3),
            "hook_strength": sc.get("hook_strength", 3),
            "total": total_score,
        },
        "review_score": total_score,
        "review_feedback": content.get("feedback", ""),
        "review_pass": content.get("pass", content.get("total", 7) >= 7),
        "review_issues": content.get("issues", []),
        "review_suggestions": content.get("rewrite_suggestions", []),
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def revise_chapter(state: NovelState) -> dict:
    """Agent 角色：修订作家 → 根据审稿/连续性意见定向改稿。"""
    if MOCK:
        draft = state.get("current_draft", "")
        notes = "；".join(state.get("review_suggestions", []) or state.get("continuity_suggestions", []) or [])
        return {"current_draft": draft + f"\n\n（Mock 修订：{notes or '已根据审稿意见微调'}）"}

    issues = (state.get("review_issues", []) or []) + (state.get("continuity_issues", []) or [])
    suggestions = (state.get("review_suggestions", []) or []) + (state.get("continuity_suggestions", []) or [])
    prompt = (
        "# Role: 网文修订作家\n"
        "按问题修订章节。优先修复重复剧情、换皮旧桥段和大纲偏移。\n\n"
        "## 必须修复的问题\n"
        + "\n".join(f"- {x}" for x in issues)
        + "\n\n"
        "## 修订建议\n"
        + "\n".join(f"- {x}" for x in suggestions)
        + "\n\n"
        "## 当前章节大纲锁定（必须逐项落实）\n"
        f"{_chapter_outline_lock(state)}\n\n"
        "## 必要记忆\n"
        f"{_compact_validation_context(state)}\n\n"
        "## 原稿\n"
        f"{state.get('current_draft', '')}\n\n"
        "## 输出要求\n"
        "1. 直接输出修订后的完整小说正文\n"
        "2. 不要解释修改过程\n"
        "3. 必须实质性覆盖当前章节大纲中的梗概、冲突、关键场景和结尾钩子\n"
        "4. 如果原稿和当前大纲不一致，以当前章节大纲为准重写相关段落\n"
        "5. 不得破坏剧情圣经中的人物状态、伏笔和世界规则"
    )
    draft = llm.invoke([SystemMessage(content=prompt)]).content
    return {"current_draft": draft}


def continuity_check(state: NovelState) -> dict:
    """Agent 角色：连续性编辑 → 检查剧情圣经、人物状态和大纲一致性。"""
    if MOCK or os.getenv("NOVEL_FAST_CONTINUITY", "1") == "1":
        rep_issues, rep_suggestions = _detect_repeated_story(state)
        outline_issues, outline_suggestions = _outline_alignment_issues(state)
        issues = list(dict.fromkeys(rep_issues + outline_issues))
        suggestions = list(dict.fromkeys(rep_suggestions + outline_suggestions))
        return {
            "continuity_pass": not issues,
            "continuity_issues": issues,
            "continuity_suggestions": suggestions,
        }

    prompt = (
        "# Role: 连续性编辑 / 设定监督\n"
        "你负责检查章节是否违背剧情圣经、篇章目标、节奏段目标、人物状态、世界规则和当前章节大纲。\n\n"
        "## 当前章节大纲锁定\n"
        f"{_compact_validation_context(state)}\n\n"
        "## 待检查章节\n"
        f"{state.get('current_draft', '')}\n\n"
        "## 重复剧情检查\n"
        "如果当前章节与前几章在事件推进、冲突结构、场景或对话上高度重复，必须判为不通过。\n\n"
        "## 输出 JSON\n"
        "{\n"
        '  "pass": true,\n'
        '  "issues": ["连续性问题1"],\n'
        '  "rewrite_suggestions": ["修复建议1"]\n'
        "}\n\n"
        "判定标准：只要出现设定冲突、人物动机断裂、伏笔遗忘、章节钩子没有落实、与当前章节大纲不一致、重复前文剧情或重复桥段，则 pass=false。"
    )
    raw_content = llm.invoke([SystemMessage(content=prompt)]).content
    try:
        content = _parse_json(raw_content)
    except ValueError:
        return {
            "continuity_pass": False,
            "continuity_issues": [f"连续性检查返回未按JSON格式输出，原始意见：{_compact_text(raw_content)}"],
            "continuity_suggestions": ["请重新生成本章，避免和前文重复。"],
        }
    rep_issues, rep_suggestions = _detect_repeated_story(state)
    outline_issues, outline_suggestions = _outline_alignment_issues(state)
    issues = list(content.get("issues", []))
    suggestions = list(content.get("rewrite_suggestions", content.get("suggestions", [])))
    for item in rep_issues:
        if item not in issues:
            issues.append(item)
    for item in rep_suggestions:
        if item not in suggestions:
            suggestions.append(item)
    for item in outline_issues:
        if item not in issues:
            issues.append(item)
    for item in outline_suggestions:
        if item not in suggestions:
            suggestions.append(item)
    return {
        "continuity_pass": content.get("pass", True) and not rep_issues and not outline_issues,
        "continuity_issues": issues,
        "continuity_suggestions": suggestions,
    }


def update_story_bible(state: NovelState) -> dict:
    """Agent 角色：剧情档案员 → 归档章节事实、伏笔和人物状态。"""
    bible = _normalize_story_bible(
        state.get("story_bible", {}),
        state.get("world_setting", {}),
        state.get("characters", []),
    )
    idx = state.get("current_chapter_idx", 0)
    chapter = (state.get("outline", []) or [{}])[idx] if idx < len(state.get("outline", []) or []) else {}

    if MOCK or llm is None or os.getenv("NOVEL_FAST_MEMORY", "1") == "1":
        summary = f"第{chapter.get('chapter_number', idx + 1)}章：{chapter.get('summary', '')}"
        bible["timeline"] = list(dict.fromkeys((bible.get("timeline", []) or []) + [summary]))[-200:]
        bible["last_chapter_summary"] = summary
        hook = (chapter.get("hook", "") or "").strip()
        if hook:
            bible["open_threads"] = list(dict.fromkeys((bible.get("open_threads", []) or []) + [hook]))[-50:]
        event = _chapter_event_record({"outline": state.get("outline", [])}, idx, state.get("current_draft", ""), {"review_scores": {"total": 0}, "continuity_pass": True})
        bible["chapter_events"] = (list(bible.get("chapter_events", []) or []) + [event])[-200:]
        bible["blacklist"] = (list(bible.get("blacklist", []) or []) + [_event_blacklist_entry(event)])[-200:]
        return {"story_bible": bible}

    prompt = (
        "# Role: 剧情档案员\n"
        "你负责把已通过审稿和连续性检查的章节更新进剧情圣经。"
        "只记录明确发生的事实，不要编造未出现内容。\n\n"
        "## 旧剧情圣经\n"
        f"{json.dumps(bible, ensure_ascii=False)}\n\n"
        "## 当前章节大纲\n"
        f"{json.dumps(chapter, ensure_ascii=False)}\n\n"
        "## 当前章节正文\n"
        f"{state.get('current_draft', '')}\n\n"
        "## 输出 JSON\n"
        "只输出一个 JSON 对象，不要使用 Markdown 代码块，不要添加“好的”等说明文字。\n"
        "{\n"
        '  "timeline": ["已发生事件"],\n'
        '  "character_states": [{"name":"角色名","status":"当前状态","goal":"当前目标","relationship_changes":"关系变化"}],\n'
        '  "open_threads": ["仍未回收的伏笔"],\n'
        '  "resolved_threads": ["本章回收的伏笔"],\n'
        '  "locations": ["地点或重要物件状态"],\n'
        '  "continuity_notes": "后续必须遵守的设定备注",\n'
        '  "last_chapter_summary": "本章100字以内摘要"\n'
        "}\n"
    )
    raw_content = llm.invoke([SystemMessage(content=prompt)]).content
    try:
        content = _parse_json(raw_content)
    except ValueError:
        chapter_no = chapter.get("chapter_number", idx + 1)
        summary = chapter.get("summary") or state.get("current_draft", "").strip().replace("\n", " ")[:120]
        summary = f"第{chapter_no}章：{summary}"
        bible["timeline"] = list(bible.get("timeline", []) or []) + [summary]
        bible["last_chapter_summary"] = summary
        hook = (chapter.get("hook") or "").strip()
        if hook:
            bible["open_threads"] = list(dict.fromkeys((bible.get("open_threads", []) or []) + [hook]))[-20:]
        note = f"第{chapter_no}章剧情圣经自动更新未能解析AI返回，已保留章节摘要。"
        old_notes = (bible.get("continuity_notes") or "").strip()
        bible["continuity_notes"] = f"{old_notes}\n{note}".strip()
        return {"story_bible": bible}
    chapter_events = list(bible.get("chapter_events", []) or [])
    event = _chapter_event_record({"outline": state.get("outline", [])}, idx, state.get("current_draft", ""), {"review_scores": {"total": 0}, "continuity_pass": True})
    chapter_events.append(event)
    safe = _safe_story_bible(content, bible, chapter, state.get("current_draft", ""), _event_blacklist_entry(event))
    safe["chapter_events"] = chapter_events[-200:]
    return {"story_bible": safe}


def advance_chapter(state: NovelState) -> dict:
    """保存章节，推进索引"""
    completed = list(state.get("completed_chapters", []))
    completed.append(state["current_draft"])

    next_idx = state["current_chapter_idx"] + 1
    total = len(state.get("outline", []))

    # 最后一步构建终稿摘要
    final_novel = None
    if next_idx >= total:
        final_novel = "\n\n---\n\n".join(completed)

    return {
        "completed_chapters": completed,
        "current_chapter_idx": next_idx,
        "rewrite_count": 0,
        "archived_draft": state["current_draft"],
        "current_draft": "",
    }


# ═══════════════════════════════════════════════
# 3. 路由
# ═══════════════════════════════════════════════

def decide_after_review(state: NovelState) -> Literal["continuity", "revise", "force_continuity"]:
    if state["review_pass"]:
        return "continuity"
    if state["rewrite_count"] < 3:
        return "revise"
    return "force_continuity"


def decide_after_continuity(state: NovelState) -> Literal["memory", "revise", "force_memory"]:
    if state.get("continuity_pass", True):
        return "memory"
    if state.get("rewrite_count", 0) < 3:
        return "revise"
    return "force_memory"


def decide_after_advance(state: NovelState) -> Literal["continue", "__end__"]:
    if state["current_chapter_idx"] >= len(state.get("outline", [])):
        return END
    return "continue"


# ═══════════════════════════════════════════════
# 4. Mock 数据
# ═══════════════════════════════════════════════

def _mock_outline() -> dict:
    return {
        "world_setting": {
            "name": "诸天万界商城",
            "genre": "都市玄幻",
            "premise": "一个普通高中生获得了一个可以兑换万界物品的商城系统，但每次消费都会引来追杀。",
            "rules": "系统强制消费，积分可兑换任何能力或物品。兑换时产生时空波动，引来源自异次元的猎杀者。猎杀者击杀宿主可获得系统碎片。",
            "locations": [
                "青城一中：主角就读的普通高中",
                "旧城区废墟：主角第一次遭遇猎杀者的地点",
                "商城虚空空间：系统内部的神秘空间"
            ],
            "factions": [
                "系统本身：神秘的存在，似乎有自我意志",
                "猎杀者：异次元的追猎者，被积分波动吸引",
                "普通人类：对超自然世界一无所知"
            ],
            "power_system": "诸天商城系统——消耗积分兑换来自不同位面的能力（漫威超能力、修仙功法、科技装备等）。积分通过完成任务或达成特定条件获得。兑换等级越高，引来的猎杀者越强。",
            "tone": "轻松幽默为主，战斗场面稍微紧张"
        },
        "volume_outline": {
            "name": "第一卷：觉醒篇",
            "arc": "林辰从被迫激活系统的普通高中生，逐步摸清系统规则与猎杀者的存在，在一系列险境中成长，最终决定主动出击查明系统真相。",
            "chapter_start": 1,
            "chapter_end": 40,
            "core_conflict": "被系统强制消费引来猎杀 vs 想低调求生",
            "climax": "林辰识破系统的部分阴谋，第一次主动反杀高阶猎杀者。",
            "key_events": ["系统激活与首次遭遇猎杀者", "被迫兑换能力打赢第一战", "发现积分消费与猎杀强度的关联", "猎杀者透露系统真相碎片", "主动反击，觉醒决心"]
        },
        "characters": [
            {
                "name": "林辰",
                "role": "主角",
                "archetype": "谨慎型",
                "appearance": "17岁，普通的黑色短发，身材偏瘦",
                "personality": "性格谨慎，遇事先想退路。但关键时刻能鼓起勇气。",
                "background": "普通家庭出身，成绩中上，没有特别突出的地方。",
                "motivation": "活下去，保护身边人。",
                "abilities": "诸天商城系统的绑定者，可兑换万界能力。",
                "flaws": "过度谨慎，容易犹豫不决错失良机。",
                "arc": "从胆小怕事的高中生成长为敢于直面危险的觉醒者。"
            },
            {
                "name": "系统",
                "role": "关键配角",
                "archetype": "腹黑型",
                "appearance": "没有实体，在意识中显示为全息界面",
                "personality": "表面机械公事，实际上有自己的盘算。",
                "background": "来历不明的远古造物。",
                "motivation": "目的不明，似乎在利用宿主达成某种目标。",
                "abilities": "跨越位面进行交易和兑换，屏蔽感知。",
                "flaws": "受规则限制，不能直接帮助宿主。",
                "arc": "从神秘工具逐渐暴露出真实目的。"
            }
        ],
        "pacing_outline": [
            {
                "name": "开局钩子段",
                "chapter_start": 1,
                "chapter_end": 5,
                "purpose": "激活系统、抛出追杀危机并建立主角求生目标。",
                "major_turning_point": "主角意识到系统既是金手指也是危险源。",
                "tension_goal": "用连续危机让读者追读。"
            },
            {
                "name": "规则探索段",
                "chapter_start": 6,
                "chapter_end": 15,
                "purpose": "探索积分、兑换和猎杀强度之间的因果。",
                "major_turning_point": "主角第一次主动设计兑换策略。",
                "tension_goal": "爽点和代价同步升级。"
            },
            {
                "name": "反派压迫段",
                "chapter_start": 16,
                "chapter_end": 30,
                "purpose": "揭开猎杀者组织与系统碎片秘密。",
                "major_turning_point": "主角遭遇高阶猎杀者并阶段性失败。",
                "tension_goal": "持续加压，扩大世界观。"
            },
            {
                "name": "篇章高潮段",
                "chapter_start": 31,
                "chapter_end": 40,
                "purpose": "收束觉醒篇核心矛盾，完成第一次主动反击。",
                "major_turning_point": "主角识破系统部分阴谋并反杀强敌。",
                "tension_goal": "强高潮后留下更大谜题。"
            }
        ],
        "story_bible": {
            "timeline": [],
            "character_states": [
                {"name": "林辰", "status": "普通高中生，刚绑定诸天商城系统。", "goal": "活下去并保护身边人。", "relationship_changes": ""},
                {"name": "系统", "status": "来历不明，表面机械，疑似有自我意志。", "goal": "目的不明。", "relationship_changes": ""}
            ],
            "open_threads": ["系统真实目的", "猎杀者为何能感知兑换波动"],
            "resolved_threads": [],
            "locations": ["青城一中", "旧城区废墟", "商城虚空空间"],
            "continuity_notes": "兑换会产生时空波动并引来猎杀者；系统受规则限制不能直接救主角。",
            "last_chapter_summary": ""
        },
        "outline": [
            {
                "chapter_number": 1,
                "title": "天降系统",
                "summary": "林辰在放学路上意外激活系统，被猎杀者盯上。",
                "conflict": "系统强制新手任务 vs 林辰想低调的愿望",
                "hook": "猎杀者已经来到学校门口。",
                "key_scenes": ["系统激活", "强制新手任务", "猎杀者降临"],
                "characters_involved": ["林辰", "系统"]
            },
            {
                "chapter_number": 2,
                "title": "首战",
                "summary": "林辰被迫兑换能力，与猎杀者正面交锋。",
                "conflict": "能力不足 vs 猎杀者强大",
                "hook": "系统宣布：一周内必须消费100积分，否则抹杀。",
                "key_scenes": ["兑换能力", "与猎杀者战斗", "系统新任务"],
                "characters_involved": ["林辰", "系统"]
            },
            {
                "chapter_number": 3,
                "title": "积分危机",
                "summary": "林辰在消费与生存之间做出危险的选择。",
                "conflict": "消费积分引来更强敌人 vs 不消费被系统抹杀",
                "hook": "他兑换了远超自己承受能力的高级能力。",
                "key_scenes": ["研究积分规则", "冒险消费", "高级猎杀者出现"],
                "characters_involved": ["林辰", "系统"]
            },
            {
                "chapter_number": 4,
                "title": "系统的秘密",
                "summary": "猎杀者透露了系统的部分真相。",
                "conflict": "相信系统 vs 相信猎杀者的警告",
                "hook": "系统的界面出现了一行从未见过的红字。",
                "key_scenes": ["猎杀者的谈判", "真相碎片", "系统异常"],
                "characters_involved": ["林辰", "系统"]
            },
            {
                "chapter_number": 5,
                "title": "绝地反击",
                "summary": "林辰在绝境中做出最终选择。",
                "conflict": "保守求生 vs 冒险反击",
                "hook": "系统露出微笑：'你终于做对了一次选择。'",
                "key_scenes": ["积分告急", "最终抉择", "反击战"],
                "characters_involved": ["林辰", "系统"]
            }
        ],
        "completed_chapters": [],
        "current_chapter_idx": 0,
        "rewrite_count": 0,
    }


def _mock_chapter(idx: int, chapter: dict) -> dict:
    return {
        "current_draft": (
            f"\u3010\u7b2c{idx+1}\u7ae0 {chapter.get('title','')}\u3011\n\n"
            f"\u6797\u8fb0\u6309\u7167\u5927\u7eb2\u300c{chapter.get('summary','')}\u300d\u5c55\u5f00\u4e86\u884c\u52a8\u3002\n"
            f"\u5f53\u524d\u51b2\u7a81\uff1a{chapter.get('conflict','')}\n"
            f"\u7ed3\u5c3e\u94a9\u5b50\uff1a{chapter.get('hook','')}\n\n"
            "\uff08Mock \u6a21\u5f0f\u2014\u53d6\u6d88\u52fe\u9009 Mock \u5373\u53ef\u8dd1\u771f\u5b9e\u5185\u5bb9\uff09\n" * 50
        )
    }


def _mock_review() -> dict:
    return {
        "review_scores": {
            "logic_consistency": 4,
            "pacing": 4,
            "character_consistency": 4,
            "language_quality": 3,
            "hook_strength": 4,
            "total": 7,
        },
        "review_feedback": "\u8282\u594f\u4e0d\u9519\uff0c\u5bf9\u8bdd\u81ea\u7136\u3002\u8bed\u8a00\u4e0a\u5076\u6709\u201cAI\u5473\u201d\uff0c\u5efa\u8bae\u51cf\u5c11\u201c\u4ed6\u60f3\u9053\u201d\u201c\u4ed6\u89c9\u5f97\u201d\u7b49\u5185\u5fc3\u72ec\u767d\u3002",
        "review_pass": True,
        "review_issues": ["\u8bed\u8a00\u5076\u6709AI\u5473", "\u67d0\u4e9b\u5904\u53d9\u8ff0\u8fc7\u4e8e\u76f4\u767d"],
        "review_suggestions": ["\u51cf\u5c11\u5185\u5fc3\u72ec\u767d\uff0c\u591a\u7528\u5bf9\u8bdd\u548c\u884c\u52a8\u8868\u8fbe"],
        "rewrite_count": 1,
    }


# ═══════════════════════════════════════════════
# 5. JSON 解析
# ═══════════════════════════════════════════════

def _parse_json(text: str) -> dict:
    text = (text or "").strip().lstrip("\ufeff")
    errors = []

    def try_load(candidate: str):
        candidate = candidate.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            return None

    parsed = try_load(text)
    if parsed is not None:
        return parsed

    fenced_blocks = re.findall(r"```(?:json|JSON)?\s*([\s\S]*?)```", text)
    for block in fenced_blocks:
        parsed = try_load(block)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        candidate = text[match.start():]
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return parsed
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    start = text.find("{")
    if start == -1:
        start = text.find("[")
    end = max(text.rfind("}"), text.rfind("]"))
    if start != -1 and end != -1 and end > start:
        parsed = try_load(text[start : end + 1])
        if parsed is not None:
            return parsed

    detail = errors[-1] if errors else "no JSON object found"
    raise ValueError(f"Cannot parse JSON ({detail}):\n{text[:500]}")


# ═══════════════════════════════════════════════
# 6. Build
# ═══════════════════════════════════════════════

def _add_chapter_nodes(builder: StateGraph) -> None:
    builder.add_node("write", write_chapter)
    builder.add_node("review", review_chapter)
    builder.add_node("revise", revise_chapter)
    builder.add_node("continuity", continuity_check)
    builder.add_node("memory", update_story_bible)
    builder.add_node("advance", advance_chapter)

    builder.add_edge("write", "review")
    builder.add_edge("revise", "review")
    builder.add_conditional_edges(
        "review",
        decide_after_review,
        {"continuity": "continuity", "revise": "revise", "force_continuity": "continuity"},
    )
    builder.add_conditional_edges(
        "continuity",
        decide_after_continuity,
        {"memory": "memory", "revise": "revise", "force_memory": "memory"},
    )
    builder.add_edge("memory", "advance")


def build_chapter_pipeline():
    builder = StateGraph(NovelState)
    builder.add_edge(START, "write")
    _add_chapter_nodes(builder)
    return builder.compile(checkpointer=MemorySaver())


def run_single_chapter(initial_state: dict) -> dict:
    """从已有大纲生成当前章节，返回 LangGraph 最终状态。"""
    app = build_chapter_pipeline()
    config = {"configurable": {"thread_id": f"single-{initial_state.get('current_chapter_idx', 0)}"}}
    last = dict(initial_state)
    for event in app.stream(initial_state, config):
        for value in event.values():
            last.update(value)
    final = app.get_state(config)
    if final and final.values:
        last.update(final.values)
    return last


def run_full_generation(initial_state: dict) -> list:
    """从已有大纲开始，运行 write -> review -> continuity -> memory -> advance 循环。"""
    builder = StateGraph(NovelState)
    builder.add_edge(START, "write")
    _add_chapter_nodes(builder)
    builder.add_conditional_edges(
        "advance", decide_after_advance,
        {"continue": "write", END: END},
    )
    app = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "full-gen"}}
    chapters = []
    for event in app.stream(initial_state, config):
        for n, v in event.items():
            if n == "advance":
                chapters = v.get("completed_chapters", [])
    return chapters

def build_pipeline():
    builder = StateGraph(NovelState)

    builder.add_node("outline", generate_outline)
    _add_chapter_nodes(builder)

    builder.add_edge(START, "outline")
    builder.add_edge("outline", "write")

    builder.add_conditional_edges(
        "advance",
        decide_after_advance,
        {"continue": "write", END: END},
    )

    return builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════
# 7. Graph
# ═══════════════════════════════════════════════

def show_graph():
    app = build_pipeline()
    graph = app.get_graph()
    print("=" * 60)
    print("  Workflow Graph (Mermaid)")
    print("=" * 60)
    print(graph.draw_mermaid())
    print()
    if EXPORT_GRAPH:
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        png_path = out_dir / "graph.png"
        png_path.write_bytes(graph.draw_mermaid_png())
        print(f"  [Saved] outputs/graph.png")


# ═══════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    if SHOW_GRAPH or EXPORT_GRAPH:
        show_graph()
        sys.exit(0)

    if not MOCK and llm is None:
        print("[!] No valid OPENAI_API_KEY found.")
        print("    Edit .env or use --mock")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    idea = (
        "\u4e00\u4e2a\u666e\u901a\u9ad8\u4e2d\u751f\u610f\u5916\u7ed1\u5b9a"
        "\u201c\u8bf8\u5929\u4e07\u754c\u5546\u57ce\u7cfb\u7edf\u201d\uff0c"
        "\u80fd\u7528\u79ef\u5206\u5151\u6362\u6f2b\u5a01\u8d85\u80fd\u529b\uff0c"
        "\u4f46\u6bcf\u6b21\u5151\u6362\u90fd\u4f1a\u5f15\u6765\u5f02\u6b21\u5143\u602a\u7269\u8ffd\u6740\u3002"
        "\u4ed6\u60f3\u4f4e\u8c03\u53d1\u80b2\uff0c\u7cfb\u7edf\u5374\u5f3a\u5236"
        "\u4ed6\u201c\u4e00\u5468\u5185\u5fc5\u987b\u6d88\u8d39100\u79ef\u5206\uff0c\u5426\u5219\u62b9\u6740\u201d\u3002"
    )
    if args:
        idea = " ".join(args)

    print("=" * 60)
    print("  [Novel v2] \u7f51\u6587\u751f\u6210\u6d41\u6c34\u7ebf  (LangGraph)")
    print("=" * 60)
    print(f"  Idea: {idea[:80]}...")
    print(f"  Mode: {'mock' if MOCK else os.getenv('NOVEL_LLM_MODEL', 'gpt-4o-mini')}")
    print()

    app = build_pipeline()
    config = {"configurable": {"thread_id": "novel-run-001"}}
    outline_list = []

    for event in app.stream({"user_idea": idea}, config):
        for name, value in event.items():
            tag = {"outline": "Outline", "write": "Write", "review": "Review", "advance": "Archive"}.get(name, name)
            if name == "outline":
                ws = value.get("world_setting", {})
                chars = value.get("characters", [])
                ol = value.get("outline", [])
                outline_list = ol
                print(f"  [{tag}] \u4e16\u754c\u89c2: {ws.get('name','')} ({ws.get('genre','')})")
                for c in chars:
                    print(f"        \u89d2\u8272: {c.get('name','')} - {c.get('archetype','')} / {c.get('motivation','')}")
                for ch in ol:
                    print(f"        \u7b2c{ch.get('chapter_number','?')}\u7ae0 {ch.get('title','')}: {ch.get('conflict','')} \u2192 {ch.get('hook','')}")
            elif name == "write":
                draft = value.get("current_draft", "")
                print(f"  [{tag}] \u8349\u7a3f\u5b8c\u6210  ({len(draft)} \u5b57)")
            elif name == "review":
                sc = value.get("review_scores", {})
                print(f"  [{tag}] \u903b\u8f91{sc.get('logic_consistency','?')}/5 "
                      f"\u8282\u594f{sc.get('pacing','?')}/5 "
                      f"\u8bed\u8a00{sc.get('language_quality','?')}/5 "
                      f"\u94a9\u5b50{sc.get('hook_strength','?')}/5 "
                      f"| \u603b\u5206{sc.get('total','?')}/10 "
                      f"| \u901a\u8fc7:{value.get('review_pass','?')} "
                      f"\u91cd\u5199{value.get('rewrite_count',0)}/3")
                issues = value.get("review_issues", [])
                if issues:
                    print(f"        \u95ee\u9898: {'; '.join(issues)}")
            elif name == "advance":
                chapters = value.get("completed_chapters", [])
                n = value.get("current_chapter_idx", 0)
                total = len(outline_list)
                print(f"  [{tag}] \u7b2c {n}\u7ae0\u5b8c\u6210  (\u5df2\u5b58 {len(chapters)} \u7ae0)")
            print()

    final = app.get_state(config)
    chapters = final.values.get("completed_chapters", [])
    print("=" * 60)
    print(f"  [Done] {len(chapters)} \u7ae0\u751f\u6210\u5b8c\u6210")
    print("=" * 60)
    if chapters:
        full = "\n\n---\n\n".join(chapters)
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "novel.txt"
        out_path.write_text(full, encoding="utf-8")
        print(f"  [File] {out_path}  ({len(full)} \u5b57)")

