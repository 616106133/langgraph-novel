from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class ReportState(TypedDict):
    topic: str
    outline: str
    draft: str
    reviewed: bool


def outline_node(state: ReportState) -> dict:
    """Step 1: 根据主题生成大纲"""
    topic = state["topic"]
    outline = f"{topic} 的大纲\n1. 背景介绍\n2. 核心概念\n3. 应用场景\n4. 总结"
    return {"outline": outline}


def draft_node(state: ReportState) -> dict:
    """Step 2: 根据大纲撰写草稿"""
    outline = state["outline"]
    draft = f"基于以下大纲撰写的草稿：\n{outline}\n\n（正文内容略）"
    return {"draft": draft}


def review_node(state: ReportState) -> dict:
    """Step 3: 审查草稿"""
    draft = state["draft"]
    print(f"  [审查] 收到草稿，长度 {len(draft)} 字符")
    return {"reviewed": True}


def decide_next(state: ReportState) -> Literal["__end__"]:
    """决定是否继续迭代"""
    if state["reviewed"]:
        return END
    return "draft_node"


# ---------- 构建图 ----------

builder = StateGraph(ReportState)

builder.add_node("outline_node", outline_node)
builder.add_node("draft_node", draft_node)
builder.add_node("review_node", review_node)

builder.add_edge(START, "outline_node")
builder.add_edge("outline_node", "draft_node")
builder.add_edge("draft_node", "review_node")
builder.add_conditional_edges("review_node", decide_next)

# 带内存持久化，方便查看状态
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ---------- 运行 ----------

print("=" * 50)
print("LangGraph Demo - 报告生成工作流")
print("=" * 50)

config = {"configurable": {"thread_id": "demo-001"}}

for event in graph.stream({"topic": "大语言模型"}, config):
    for node_name, value in event.items():
        print(f"\n[{node_name}]")
        for k, v in value.items():
            print(f"  {k}: {v}")

print("\n" + "=" * 50)
print("最终状态:")
print(graph.get_state(config).values)
