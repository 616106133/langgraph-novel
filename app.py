import streamlit as st
import os
import sys
import importlib
from pathlib import Path

st.set_page_config(
    page_title="LangGraph 小说流水线",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_DIR))

# Provider presets
PROVIDERS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4o-turbo"],
        "default_model": "gpt-4o-mini",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
    },
    "Custom": {
        "base_url": "",
        "models": [],
        "default_model": "",
    },
}


def _run_pipeline(idea: str, mock: bool, model: str, api_key: str, base_url: str):
    os.environ["NOVEL_MOCK"] = "1" if mock else "0"
    os.environ["NOVEL_LLM_MODEL"] = model
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    import novel_pipeline as np
    importlib.reload(np)

    app = np.build_pipeline()
    config = {"configurable": {"thread_id": "novel-run-001"}}
    chapters = []

    for event in app.stream({"user_idea": idea}, config):
        yield event
        for name, value in event.items():
            if name == "advance":
                chapters = value.get("completed_chapters", [])

    yield {"_done": True, "chapters": chapters}


def _show_graph_png():
    import novel_pipeline as np
    return np.build_pipeline().get_graph().draw_mermaid_png()


# Session state
for key in ("run_count", "chapters", "full_text", "finished", "show_graph"):
    if key not in st.session_state:
        if key == "run_count":
            st.session_state[key] = 0
        elif key == "show_graph":
            st.session_state[key] = False
        elif key in ("chapters",):
            st.session_state[key] = []
        elif key in ("full_text",):
            st.session_state[key] = ""
        else:
            st.session_state[key] = False


# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### :material/settings: 设置")

    # Provider selection
    provider = st.selectbox(
        "API 提供商",
        list(PROVIDERS.keys()),
        index=0,
        help="DeepSeek 在国内可直接访问，无需代理",
    )

    presets = PROVIDERS[provider]

    api_key = st.text_input(
        "API Key",
        type="password",
        help=f"从 {provider} 平台获取 API Key",
    )

    model = st.selectbox(
        "模型",
        presets["models"] if presets["models"] else ["gpt-4o-mini"],
        index=0 if presets["models"] else 0,
        key="model_select",
    )

    # If custom provider, allow typing
    if provider == "Custom":
        model = st.text_input("自定义模型名", value=model, key="custom_model")

    base_url = st.text_input(
        "API Base URL",
        value=presets["base_url"],
        help="API 端点地址。使用代理时改为代理地址",
    )

    mock_mode = st.checkbox("Mock 模式（无需 Key）", value=False)

    st.divider()

    if st.button(":material/schema: 查看流程图", use_container_width=True):
        st.session_state.show_graph = not st.session_state.show_graph

    if st.button(":material/folder_open: 打开输出目录", use_container_width=True):
        import subprocess
        subprocess.Popen(["explorer", str(OUTPUT_DIR.resolve())])

    st.divider()
    st.caption(f"工作目录：`{PROJECT_DIR}`")

    # Quick tips
    with st.expander(":material/lightbulb: 连接问题"):
        st.markdown(
            """
- **国内用 DeepSeek**：选 DeepSeek → 去 platform.deepseek.com 注册 → 免费额度 500 万 token
- **用代理**：选 Custom → 填入代理的 API 地址（如 `https://xxx.com/v1`）
- **Mock 模式**：不填 Key，勾上 Mock 即可测试流程
"""
        )


# ── Main area ──────────────────────────────────────────────

st.title(":material/menu_book: 网文生成流水线")
st.markdown("**LangGraph**  ·  4 节点线性流水线  ·  审稿重写循环（上限 3 次）")

idea = st.text_area(
    "小说创意",
    value=(
        "一个普通高中生意外绑定\u201c诸天万界商城系统\u201d，"
        "能用积分兑换漫威超能力，但每次兑换都会引来异次元怪物追杀。"
        "他想低调发育，系统却强制他\u201c一周内必须消费100积分，否则抹杀\u201d。"
    ),
    height=100,
)

col1, col2 = st.columns([2, 8])
with col1:
    run_btn = st.button(
        ":material/play_arrow: 开始生成",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.finished and st.session_state.run_count > 0,
    )


# ── Graph display ──────────────────────────────────────────

if st.session_state.show_graph:
    st.divider()
    st.markdown("### :material/schema: 工作流图")
    try:
        png = _show_graph_png()
        st.image(png, caption="LangGraph Pipeline", use_container_width=False)
    except Exception as e:
        st.code(
            "graph TD;\n"
            "    START --> outline\n"
            "    outline --> write\n"
            "    write --> review\n"
            "    review -->|pass| advance\n"
            "    review -->|fail < 3| write\n"
            "    review -->|fail >= 3| advance\n"
            "    advance -->|more| write\n"
            "    advance -->|done| FINISH",
            language="mermaid",
        )
        st.caption(f"(PNG: {e})")
    st.divider()


# ── Run ────────────────────────────────────────────────────

if run_btn and idea.strip():
    st.session_state.finished = False
    st.session_state.chapters = []
    st.session_state.full_text = ""
    st.session_state.run_count += 1

    status = st.status(":material/hourglass_top: 正在生成...", expanded=True)

    with status:
        outline_ph = st.empty()
        write_ph = st.empty()
        review_ph = st.empty()
        archive_ph = st.empty()
        error_ph = st.empty()

    try:
        for ev in _run_pipeline(idea, mock_mode, model, api_key, base_url):
            if "_done" in ev:
                st.session_state.chapters = ev["chapters"]
                st.session_state.finished = True
                continue

            for name, value in ev.items():
                if name == "outline":
                    with outline_ph.container():
                        st.markdown(f"**:material/list_alt: \u5927\u7eb2\u5df2\u751f\u6210**")
                        for i, ch in enumerate(value.get("outline", [])):
                            st.write(f"\u3000**Ch{i+1}:** {ch[:60]}")
                elif name == "write":
                    draft = value.get("current_draft", "")
                    with write_ph.container():
                        st.markdown(f"**:material/edit: \u8349\u7a3f\u5b8c\u6210** \u2014 {len(draft)} \u5b57")
                elif name == "review":
                    score = value.get("review_score", "?")
                    passed = value.get("review_pass", "?")
                    rw = value.get("rewrite_count", 0)
                    fb = value.get("review_feedback", "")
                    icon = ":material/check_circle:" if passed else ":material/error:"
                    with review_ph.container():
                        st.markdown(
                            f"**:material/rate_review: \u5ba1\u7a3f** \u2014 \u8bc4\u5206 {score}/10 | {icon} "
                            f"\u901a\u8fc7: {passed} | \u91cd\u5199 {rw}/3"
                        )
                        if fb:
                            st.write(f"\u3000*{fb[:120]}*")
                elif name == "advance":
                    chapters = value.get("completed_chapters", [])
                    n = value.get("current_chapter_idx", 0)
                    total = len(value.get("outline", []))
                    with archive_ph.container():
                        st.markdown(
                            f"**:material/save: \u5b58\u6863** \u2014 \u7b2c {n}/{total} \u7ae0\u5b8c\u6210"
                            f"\uff08\u7d2f\u8ba1 {len(chapters)} \u7ae0\uff09"
                        )

        status.update(
            label=f":material/check_circle: **\u5b8c\u6210\uff01\u5171 {len(st.session_state.chapters)} \u7ae0**",
            state="complete",
            expanded=False,
        )

        if st.session_state.chapters:
            st.session_state.full_text = "\n\n---\n\n".join(st.session_state.chapters)
            out_path = OUTPUT_DIR / "novel.txt"
            out_path.write_text(st.session_state.full_text, encoding="utf-8")

    except Exception as e:
        status.update(label=":material/error: **\u751f\u6210\u5931\u8d25**", state="error", expanded=True)
        err_msg = str(e)
        with error_ph.container():
            st.error(f"\u62b1\u6b49\uff0c\u8fd0\u884c\u65f6\u51fa\u9519\uff1a")
            st.code(err_msg[:2000], language="text")
            st.markdown(
                """
**\u53ef\u80fd\u7684\u89e3\u51b3\u65b9\u6848\uff1a**
1. \u786e\u8ba4 API Key \u662f\u5426\u6b63\u786e
2. \u8bd5\u8bd5\u5207\u6362\u5230 **DeepSeek** \u63d0\u4f9b\u5546\uff08\u56fd\u5185\u76f4\u8fde\uff09
3. \u5982\u679c\u7528\u4ee3\u7406\uff0c\u5728\u201cAPI Base URL\u201d\u6b3c\u8f93\u5165\u4ee3\u7406\u5730\u5740
4. \u6216\u76f4\u63a5\u52fe\u9009 **Mock \u6a21\u5f0f** \u6d4b\u8bd5\u6d41\u7a0b
"""
            )


# ── Results ────────────────────────────────────────────────

if st.session_state.finished and st.session_state.chapters:
    full = st.session_state.full_text

    st.success(
        f":material/check_circle: {len(st.session_state.chapters)} \u7ae0\u5df2\u751f\u6210  \u00b7  \u5171 {len(full)} \u5b57"
    )

    col_a, col_b, col_c = st.columns([2, 2, 6])
    with col_a:
        st.download_button(
            ":material/download: \u4e0b\u8f7d TXT",
            data=full.encode("utf-8"),
            file_name="novel.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_b:
        if st.button(":material/refresh: \u91cd\u65b0\u751f\u6210", use_container_width=True):
            st.session_state.finished = False
            st.session_state.chapters = []
            st.session_state.full_text = ""
            st.rerun()

    with st.expander(":material/menu_book: \u67e5\u770b\u5168\u6587", expanded=True):
        for i, ch in enumerate(st.session_state.chapters):
            st.markdown("---")
            st.markdown(f"### \u7b2c {i+1} \u7ae0")
            st.markdown(ch[:3000])
            if len(ch) > 3000:
                st.info(f"...\uff08\u5269\u4f59 {len(ch)-3000} \u5b57\uff09")
