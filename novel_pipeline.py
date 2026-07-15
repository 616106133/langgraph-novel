# -*- coding: utf-8 -*-
"""
Novel Writing Pipeline -- LangGraph MVP
v2: structured data + refined prompts
"""

import json
import os
import sys
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


class NovelState(TypedDict):
    # 输入
    user_idea: str

    # 角色状态
    character_states: list = []

    # 大纲阶段
    world_setting: WorldSetting
    characters: List[CharacterCard]
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
    rewrite_count: int


# ═══════════════════════════════════════════════
# 2. 节点函数
# ═══════════════════════════════════════════════


def generate_outline(state: NovelState) -> dict:
    """Agent 角色：创意架构师 → 输出结构化世界观 + 角色卡 + 大纲"""
    if MOCK:
        return _mock_outline()

    prompt = (
        "# Role: 资深网文编辑 / 创意架构师\n\n"
        "你需要根据用户的一句话创意，设计一套完整的小说起始设定。\n"
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
        "- 每章必须有一个明确的冲突和一个让人想读下一章的钩子\n"
        "- 5 章构成一个完整的故事弧线（起→承→转→合→高潮）\n"
        "- 只输出 JSON，不要任何额外文字"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    content = _parse_json(response.content)
    return {
        "world_setting": content["world_setting"],
        "characters": content["characters"],
        "outline": content["outline"],
        "completed_chapters": [],
        "current_chapter_idx": 0,
        "rewrite_count": 0,
    }


def write_chapter(state: NovelState) -> dict:
    """Agent 角色：畅销网文作家 → 写出吸引人的章节"""
    idx = state["current_chapter_idx"]
    chapter = state["outline"][idx]

    if MOCK:
        return _mock_chapter(idx, chapter)

    prev = (
        state["completed_chapters"][-1][:200]
        if state.get("completed_chapters")
        else "\u65e0"
    )

    # 格式化角色卡供 prompt 使用
    chars_text = "\n".join(
        f"- {c['name']}（{c['role']}）: {c['personality']}。能力：{c['abilities']}。"
        for c in state["characters"]
    )

    prompt = (
        "# Role: 畅销网文作家\n"
        "你擅长用快节奏的叙事和生动的对话抓住读者，"
        "让每一章结尾都让人忍不住点开下一章。\n\n"
        "## 世界观\n"
        f"名称：{state['world_setting']['name']}\n"
        f"风格：{state['world_setting']['tone']}\n"
        f"核心规则：{state['world_setting']['rules']}\n"
        f"力量体系：{state['world_setting']['power_system']}\n\n"
        "## 角色\n"
        f"{chars_text}\n\n"
        "## 当前章节\n"
        f"第 {chapter['chapter_number']} 章：{chapter['title']}\n"
        f"梗概：{chapter['summary']}\n"
        f"核心冲突：{chapter['conflict']}\n"
        f"结尾钩子：{chapter['hook']}\n"
        f"登场角色：{', '.join(chapter['characters_involved'])}\n"
        f"关键场景：{'; '.join(chapter['key_scenes'])}\n\n"
        "## 角色状态追踪\n"
        f"{state_summary}\n\n"
        "## 前情提要（上一章末尾 200 字）\n"
        f"{prev}\n\n"
        "## 写作规范\n"
        "1. 正文 3000 字左右，节奏快慢交替\n"
        "2. **用对话推动剧情**，少用解释性旁白\n"
        "3. 结尾必须自然引出指定的钩子\n"
        "4. 确保人设一致性——角色的言行符合其性格卡\n"
        "5. 禁止出现\u201c总结一下\u201d\u201c接下来\u201d\u201c如前所述\u201d等元叙述词\n"
        "6. 直接输出小说正文，不要任何标题或说明文字"
    )
    draft = llm.invoke([SystemMessage(content=prompt)]).content
    return {"current_draft": draft}


def review_chapter(state: NovelState) -> dict:
    """Agent 角色：审稿编辑 → 结构化评分 + 具体修改建议"""
    if MOCK:
        return _mock_review()

    prompt = (
        "# Role: 资深网文审稿编辑\n"
        "你在网文平台工作 10 年，眼光毒辣，\n"
        "能从读者视角准确判断一个章节的质量。\n\n"
        "## 评审维度（每项 1-5 分）\n\n"
        "1. **logic_consistency**（逻辑一致性）\n"
        "   - 5: 角色言行完全符合人设，情节推进合理\n"
        "   - 3: 有小瑕疵但不影响阅读\n"
        "   - 1: 出现明显 OOC（人设崩塌）或逻辑硬伤\n\n"
        "2. **pacing**（节奏张力）\n"
        "   - 5: 张弛有度，层层递进，让人想一口气读完\n"
        "   - 3: 有拖沓或仓促感，但整体可接受\n"
        "   - 1: 流水账式叙述，或情节跳跃混乱\n\n"
        "3. **language_quality**（语言质量）\n"
        "   - 5: 生动自然，对话贴近角色性格\n"
        "   - 3: 偶有突兀的书面语或\u201cAI 味\u201d表达\n"
        "   - 1: 大量元叙述、强行总结、情感直白\n\n"
        "4. **hook_strength**（钩子强度）\n"
        "   - 5: 结尾让人迫切想知道后续\n"
        "   - 3: 有悬念但不够强烈\n"
        "   - 1: 平铺直叙，没有吸引继续读的动力\n\n"
        "## 输出 JSON（严格遵循）\n"
        "```json\n"
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
        "```\n"
        "判定标准：total >= 7 且无致命问题则 pass = true\n\n"
        "## 待审章节\n"
        f"{state['current_draft']}\n\n"
        "直接输出 JSON，不要任何额外文字。"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    content = _parse_json(response.content)

    sc = content["scores"]
    return {
        "review_scores": {
            "logic_consistency": sc.get("logic_consistency", 3),
            "pacing": sc.get("pacing", 3),
            "character_consistency": sc.get("character_consistency", 3),
            "language_quality": sc.get("language_quality", 3),
            "hook_strength": sc.get("hook_strength", 3),
            "total": content.get("total", sum(sc.values()) // 2),
        },
        "review_feedback": content.get("feedback", ""),
        "review_pass": content.get("pass", content.get("total", 7) >= 7),
        "review_issues": content.get("issues", []),
        "review_suggestions": content.get("rewrite_suggestions", []),
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


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
        "current_draft": "",
    }


# ═══════════════════════════════════════════════
# 3. 路由
# ═══════════════════════════════════════════════

def decide_after_review(state: NovelState) -> Literal["advance", "rewrite", "force_advance"]:
    if state["review_pass"]:
        return "advance"
    if state["rewrite_count"] < 3:
        return "rewrite"
    return "force_advance"


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
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON:\n{text[:500]}")


# ═══════════════════════════════════════════════
# 6. Build
# ═══════════════════════════════════════════════

# 在 build_pipeline() 之前插入
def run_full_generation(initial_state: dict) -> List[str]:
    """从已有大纲开始，运行 write -> review -> advance 循环，返回 completed_chapters。"""
    import importlib as _il
    _il.reload(sys.modules[__name__])

    builder = StateGraph(NovelState)

    builder.add_node("write", write_chapter)
    builder.add_node("review", review_chapter)
    builder.add_node("advance", advance_chapter)

    builder.add_edge(START, "write")
    builder.add_edge("write", "review")

    builder.add_conditional_edges(
        "review",
        decide_after_review,
        {"advance": "advance", "rewrite": "write", "force_advance": "advance"},
    )

    builder.add_conditional_edges(
        "advance",
        decide_after_advance,
        {"continue": "write", END: END},
    )

    app = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "full-gen"}}

    chapters = []
    for event in app.stream(initial_state, config):
        for name, value in event.items():
            if name == "advance":
                chapters = value.get("completed_chapters", [])
    return chapters



def run_full_generation(initial_state: dict) -> list:
    """从已有大纲开始，运行 write -> review -> advance 循环，返回 chapters。"""
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    builder = StateGraph(NovelState)
    builder.add_node("write", write_chapter)
    builder.add_node("review", review_chapter)
    builder.add_node("advance", advance_chapter)
    builder.add_edge(START, "write")
    builder.add_edge("write", "review")
    builder.add_conditional_edges(
        "review", decide_after_review,
        {"advance": "advance", "rewrite": "write", "force_advance": "advance"},
    )
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
    builder.add_node("write", write_chapter)
    builder.add_node("review", review_chapter)
    builder.add_node("advance", advance_chapter)

    builder.add_edge(START, "outline")
    builder.add_edge("outline", "write")
    builder.add_edge("write", "review")

    builder.add_conditional_edges(
        "review",
        decide_after_review,
        {"advance": "advance", "rewrite": "write", "force_advance": "advance"},
    )

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

