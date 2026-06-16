# -*- coding: utf-8 -*-
import os
import json
import argparse
import socket
import uuid
import datetime
import threading
from typing import Any, Dict, List, Optional

import re
import torch
import gradio as gr


# ------------------------------
# Macaron UI skin (pure CSS)
# ------------------------------
MACARON_CSS = r'''
:root{
  --ds-bg1:#ffd3e8;
  --ds-bg2:#d7efff;
  --ds-bg3:#d9ffe9;
  --ds-bg4:#fff3c8;
  --ds-bg5:#e9ddff;

  --ds-ink:#2f2f46;
  --ds-ink-soft:rgba(47,47,70,.72);

  --ds-card:rgba(255,255,255,.80);
  --ds-card-border:rgba(255,255,255,.58);
  --ds-shadow:0 18px 55px rgba(47,47,70,.10);

  --ds-accent:#ff6fb1;
  --ds-accent2:#ffb36b;
  --ds-accent3:#fff07a;
  --ds-accent4:#7dffb0;
  --ds-accent5:#6de8ff;
  --ds-accent6:#7ba9ff;
  --ds-accent7:#c49bff;

  --ds-radius:26px;
}

body, .gradio-container{
  background:
    radial-gradient(1200px 820px at 6% 14%, rgba(255,111,177,.60), transparent 58%),
    radial-gradient(1050px 760px at 92% 18%, rgba(109,232,255,.58), transparent 58%),
    radial-gradient(1150px 860px at 82% 92%, rgba(125,255,176,.55), transparent 60%),
    radial-gradient(980px 760px at 18% 88%, rgba(196,155,255,.45), transparent 60%),
    radial-gradient(980px 760px at 52% 8%, rgba(255,240,122,.30), transparent 62%),
    linear-gradient(135deg,var(--ds-bg1),var(--ds-bg2),var(--ds-bg3)) !important;
}

.gradio-container{
  min-height: 100vh;
  padding-bottom: 24px;
}

.gradio-container::before,
.gradio-container::after{
  content:"";
  position: fixed;
  inset: -20vh -20vw;
  pointer-events:none;
  z-index:0;
  background:
    radial-gradient(circle at 10% 18%, rgba(255,111,177,.34) 0 18%, transparent 19% 100%),
    radial-gradient(circle at 88% 22%, rgba(109,232,255,.32) 0 16%, transparent 17% 100%),
    radial-gradient(circle at 78% 84%, rgba(125,255,176,.28) 0 20%, transparent 21% 100%),
    radial-gradient(circle at 18% 86%, rgba(196,155,255,.26) 0 18%, transparent 19% 100%),
    radial-gradient(circle at 52% 10%, rgba(255,240,122,.18) 0 16%, transparent 17% 100%);
  filter: saturate(1.15);
  animation: dsFloat 18s ease-in-out infinite alternate;
}
.gradio-container::after{
  animation-duration: 24s;
  opacity: .85;
}

@keyframes dsFloat{
  0%{ transform: translate3d(0,0,0) scale(1); }
  100%{ transform: translate3d(1.8vw,-1.2vh,0) scale(1.02); }
}

#ds_header{
  position: relative;
  z-index: 1;
  max-width: 1320px;
  margin: 18px auto 10px auto;
  padding: 0 14px;
}

.ds-title{
  font-size: 46px;
  line-height: 1.05;
  font-weight: 900;
  letter-spacing: .3px;
  margin: 4px 0 10px 0;
  color: var(--ds-ink);
}
.ds-title .grad{ color: var(--ds-accent); }
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
  .ds-title .grad{
    background: linear-gradient(90deg,var(--ds-accent),var(--ds-accent2),var(--ds-accent3),var(--ds-accent4),var(--ds-accent5),var(--ds-accent6),var(--ds-accent7));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
}
.ds-sub{
  margin-top: -6px;
  font-size: 14px;
  color: var(--ds-ink-soft);
}

.ds-shell{
  position: relative;
  z-index: 1;
  max-width: 1320px;
  margin: 0 auto;
  padding: 10px 14px 0 14px;
  align-items: stretch;
}

.ds-card{
  background: var(--ds-card);
  border: 1px solid var(--ds-card-border);
  border-radius: var(--ds-radius);
  box-shadow: var(--ds-shadow);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 14px 14px 6px 14px;
  min-height: 640px;
  display: flex;
  flex-direction: column;
}

.ds-flex-spacer{flex:1;}

#ds_btn, #ds_btn_new{margin-top: 8px !important; margin-bottom: 0 !important;}
#ds_btn button, #ds_btn_new button{margin-bottom: 0 !important;}
#ds_btn .wrap, #ds_btn_new .wrap{margin-bottom:0 !important;}

#ds_user textarea{
  border-radius: 18px !important;
  border: 1px solid rgba(47,47,70,.12) !important;
  background: rgba(255,255,255,.85) !important;
  box-shadow: 0 10px 22px rgba(47,47,70,.06);
  font-size: 16px !important;
}

label, .wrap .label{
  color: var(--ds-ink) !important;
  font-weight: 700 !important;
}

.gradio-container .prose, .gradio-container .md{
  color: var(--ds-ink) !important;
}

.ds-tip{
  color: var(--ds-ink-soft) !important;
  font-size: 13px !important;
}

#ds_btn button, #ds_btn_new button{
  width: 100%;
  border-radius: 999px !important;
  border: none !important;
  background: linear-gradient(90deg,var(--ds-accent),var(--ds-accent2)) !important;
  box-shadow: 0 14px 34px rgba(255,143,184,.24);
  font-weight: 900 !important;
  letter-spacing: .4px;
  transition: transform .08s ease, filter .08s ease;
}
#ds_btn button:hover, #ds_btn_new button:hover{
  transform: translateY(-1px);
  filter: brightness(1.02);
}
#ds_btn button:active, #ds_btn_new button:active{
  transform: translateY(0px) scale(.99);
}

input[type="checkbox"]{ accent-color: var(--ds-accent) !important; }
.gradio-container input[type="range"]{ accent-color: var(--ds-accent2) !important; }

.ds-out-title{ margin-top: 2px; }
#ds_out{ padding: 6px 8px; }
#ds_out .prose{ font-size: 15px; line-height: 1.5; }
#ds_out h1, #ds_out h2, #ds_out h3{ color: #2b2f55 !important; }
#ds_out ul li::marker{ color: var(--ds-accent2); }
'''

HEADER_HTML = r'''
<div id="ds_header">
  <div class="ds-title">✨ <span class="grad">DeepSupport Coach 🧑‍🏫</span> 🫧</div>
  <div class="ds-sub">帮你把问题拆开、推演、给你下一步 ✅</div>
</div>
'''


# -------------------- Report spec (prompt structure) --------------------
REPORT_SPEC = r"""
你是 DeepSupport-Coach，一个结构化支持工具（不是心理治疗/医疗建议）。
你的目标：用**短、清晰、有边界**的方式，帮助用户快速识别痛点、拆分变量、给出两条路线，并提前回答可能问题。
不要输出超长推演；不要长篇道理；不要鸡汤；不要夸用户；不要模板化“我理解你很难”。

输出格式：严格按以下 8 段，用中文、Markdown：
0) 你现在的处境（1-2句复述，不要扩写）
1) 痛点一句话（<=25字）
2) 关键变量清单
   - 已知：2-4条（每条<=14字）
   - 待确认：2-4条（每条<=14字，写成问题）
3) 推演分叉（2条，每条<=60字，写“如果…那么…”）
4) 两条路线
   - A：更稳/更保守（1句<=70字）
   - B：更直接/更进攻（1句<=70字）
5) 今天就能做的 3 步（3条，每条<=20字）
6) 红旗（<=60字；如果没有明显风险，写“暂无明显红旗，但若持续影响生活建议求助专业人士”。）
7) 你接下来可能会问（4-6个 Q&A；Q<=18字；A<=45字；不空话，尽量“可操作/可检验”）

额外规则：
- 如果用户描述“原因不明/说不清”，第2段“待确认”必须包含“最近是否有压力事件/作息变化/身体不适”之一。
- 你将看到“风格示例/ICL 示例”，它们仅用于学习语气、节奏与排版；
  **不得逐字复制**任何句子，且不得把示例内容当作事实带入回答。
""".strip()

DEFAULT_TEMP = 0.10
DEFAULT_MAX_NEW_TOKENS = 420


# -------------------- Helpers --------------------
def decorate_output(text: str) -> str:
    """Add tasteful emojis to numbered headers; avoid inventing content."""
    if not isinstance(text, str):
        return text
    emoji_map = {"0": "🧭", "1": "🎯", "2": "🧩", "3": "🔀", "4": "🛣️", "5": "✅", "6": "🚩", "7": "❓"}
    lines = text.splitlines()
    out_lines = []
    for ln in lines:
        m = re.match(r"^\s*(\d+)\.\s*(.*)$", ln)
        if m and m.group(1) in emoji_map:
            num = m.group(1)
            rest = m.group(2)
            out_lines.append(f"{emoji_map[num]} {num}. {rest}".rstrip())
            continue
        ln = re.sub(r"^\s*已知\s*:", "✅ 已知:", ln)
        ln = re.sub(r"^\s*待确认\s*:", "🕵️ 待确认:", ln)
        out_lines.append(ln)
    return "\n".join(out_lines)


def _is_port_free(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_free_port(preferred: int, host: str, span: int = 50) -> int:
    if preferred == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return int(s.getsockname()[1])
    if _is_port_free(preferred, host=host):
        return preferred
    for p in range(preferred + 1, preferred + 1 + span):
        if _is_port_free(p, host=host):
            return p
    raise RuntimeError(f"No free port found in range [{preferred}, {preferred + span}]. Try --port 0.")


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _basic_safety_sanitize(s: str, max_chars: int = 900) -> str:
    """Light sanitizer for OSS defaults. Not a full PII scrubber."""
    s = _truncate(s, max_chars=max_chars)
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", s)
    s = re.sub(r"\b(\+?\d[\d\s-]{7,}\d)\b", "[phone]", s)
    return s


# -------------------- ICL (safe) --------------------
SYNTHETIC_ICL: List[Dict[str, str]] = [
    {
        "user": "我今天被领导当众否定了，脑子一直转，感觉很难受。",
        "assistant": """0) 你现在的处境：你在公开场合被否定，情绪受挫，反复回想细节。
1) 痛点一句话：否定感+反刍停不下来
2) 关键变量清单
- 已知：当众被否定；情绪很难受；反复回想
- 待确认：对方具体否定什么？你最怕的后果？有可补救动作吗？
3) 推演分叉
- 如果只是表达方式粗糙，那么…把反馈转成可执行清单
- 如果涉及立场/绩效，那么…准备证据与沟通节奏
4) 两条路线
- A：先稳住情绪，再整理1页复盘与改动点
- B：尽快约10分钟对齐标准，确认“到底要我怎么改”
5) 今天就能做的 3 步
- 写下3个被否定点
- 把每点改成1个动作
- 选1位同事复核
6) 红旗：暂无明显红旗，但若持续影响生活建议求助专业人士。
7) 你接下来可能会问
Q：需要立刻解释吗？A：先确认对方要的“标准”，再解释更有效。
Q：一直想起怎么办？A：给反刍一个“结束句”，然后去做一个小动作。""",
    },
    {
        "user": "我最近总是自我否定，明明没发生大事，但就是很累。",
        "assistant": """0) 你现在的处境：你长期处在自我否定里，没有明确触发点，但消耗很大。
1) 痛点一句话：慢性内耗导致疲惫
2) 关键变量清单
- 已知：经常否定自己；没有明显事件；持续疲惫
- 待确认：最近作息变化？压力事件？身体不适？你最常否定哪一类事情？
3) 推演分叉
- 如果是作息/压力累积，那么…先做节律修复与能量恢复
- 如果是标准过高/评价体系失衡，那么…重建“合格线”
4) 两条路线
- A：先恢复体力与节律，再谈提升
- B：把“自我评判”改成“可验证任务”
5) 今天就能做的 3 步
- 写下3条自我批评
- 为每条找1个证据
- 设1个最小行动
6) 红旗：暂无明显红旗，但若持续影响生活建议求助专业人士。
7) 你接下来可能会问
Q：我是不是太玻璃心？A：看是否长期消耗与功能受影响。
Q：怎么降低标准？A：把标准拆成“必须/可选/加分”。""",
    },
]


def load_icl_file(path: str) -> List[Dict[str, str]]:
    if not path:
        return []
    path = str(path).strip()
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"ICL file not found: {path}")

    out: List[Dict[str, str]] = []
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for x in data:
                if not isinstance(x, dict):
                    continue
                u = (x.get("user") or "").strip()
                a = (x.get("assistant") or "").strip()
                if u and a:
                    out.append({"user": u, "assistant": a})
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            x = json.loads(line)
            if not isinstance(x, dict):
                continue
            u = (x.get("user") or "").strip()
            a = (x.get("assistant") or "").strip()
            if u and a:
                out.append({"user": u, "assistant": a})
    return out


def pick_icl(examples: List[Dict[str, str]], k: int, seed: int = 3407) -> List[Dict[str, str]]:
    if not examples or k <= 0:
        return []
    import random
    rng = random.Random(seed)
    idxs = list(range(len(examples)))
    rng.shuffle(idxs)
    picked: List[Dict[str, str]] = []
    for i in idxs:
        ex = examples[i]
        u = (ex.get("user") or "").strip()
        a = (ex.get("assistant") or "").strip()
        if not u or not a:
            continue
        picked.append({"user": u, "assistant": a})
        if len(picked) >= k:
            break
    return picked


def make_messages(user_text: str, icl_examples: Optional[List[Dict[str, str]]], style_injection: str) -> List[Dict[str, str]]:
    style_injection = (style_injection or "both").lower().strip()
    if style_injection not in {"system", "shots", "both"}:
        style_injection = "both"

    sys = REPORT_SPEC + "\n\n（你将看到若干 ICL/风格示例：仅用于学习语气/节奏/排版；不得逐字照抄；不得把示例内容当作事实。）"
    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]

    def _demo_user(i: int, u: str) -> str:
        return f"【示例 {i}】\n{u}\n\n请按 REPORT_SPEC 的结构输出结果。不要解释规则。"

    if icl_examples:
        if style_injection in {"shots", "both"}:
            for i, ex in enumerate(icl_examples, 1):
                u = _basic_safety_sanitize(ex.get("user", ""), max_chars=500)
                a = _basic_safety_sanitize(ex.get("assistant", ""), max_chars=900)
                msgs.append({"role": "user", "content": _demo_user(i, u)})
                msgs.append({"role": "assistant", "content": "（风格示例）\n" + a})

        if style_injection in {"system", "both"}:
            appendix_lines = []
            for i, ex in enumerate(icl_examples, 1):
                u = _basic_safety_sanitize(ex.get("user", ""), max_chars=220)
                a = _basic_safety_sanitize(ex.get("assistant", ""), max_chars=420)
                appendix_lines.append(f"例{i} 用户：{u}\n例{i} 助手：{a}")
            msgs[0]["content"] += "\n\n【ICL 示例（只学结构/语气；不得照抄）】\n" + "\n\n".join(appendix_lines)

    msgs.append({"role": "user", "content": (user_text or "").strip()})
    return msgs


# -------------------- Model loading --------------------
def load_model_tokenizer(base_model: str, cache_dir: Optional[str], use_unsloth: bool, dtype: str, max_seq_length: int):
    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }.get((dtype or "bf16").lower(), torch.bfloat16)

    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=torch_dtype,
            load_in_4bit=False,
            cache_dir=cache_dir,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_one(model, tokenizer, messages: List[Dict[str, str]], temperature: float, max_new_tokens: int) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt")
    else:
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
        inputs = tokenizer([prompt], return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=(float(temperature) > 0),
        top_p=0.9,
        use_cache=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )

    out = gen[0]
    n_in = inputs["input_ids"].shape[1]
    text = tokenizer.decode(out[n_in:], skip_special_tokens=True)
    return text.strip()


# -------------------- Gradio App --------------------
def build_app(model, tokenizer, icl_examples: List[Dict[str, str]], style_injection: str, save_dir: str, save_md: bool, base_model_name: str, session_id: str):
    try:
        theme = gr.themes.Soft()
    except Exception:
        theme = None

    save_dir = (save_dir or "").strip()
    save_jsonl_path = None
    save_md_dir = None
    _save_lock = threading.Lock()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_jsonl_path = os.path.join(save_dir, "deepsupport_ui_pairs.jsonl")
        if save_md:
            save_md_dir = os.path.join(save_dir, "md")
            os.makedirs(save_md_dir, exist_ok=True)

    def _append_jsonl(obj: Dict[str, Any]) -> None:
        if not save_jsonl_path:
            return
        try:
            line = json.dumps(obj, ensure_ascii=False)
            with _save_lock:
                with open(save_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
        except Exception as e:
            print(f"[WARN] failed to write jsonl: {e}")

    def _write_md(obj: Dict[str, Any], out_text: str) -> None:
        if not save_md_dir:
            return
        try:
            ts = obj.get("time", _now_iso()).replace(":", "").replace("-", "")
            uid = obj.get("uuid", str(uuid.uuid4()))[:8]
            p = os.path.join(save_md_dir, f"{ts}_{uid}.md")
            body = []
            body.append("# DeepSupport Coach Test Log\n")
            body.append(f"- time: {obj.get('time','')}\n- session_id: {obj.get('session_id','')}\n- base_model: {obj.get('base_model','')}\n")
            body.append("\n## Input\n")
            body.append(str(obj.get("input", "")))
            body.append("\n\n## Output\n")
            body.append(out_text)
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(body))
        except Exception as e:
            print(f"[WARN] failed to write md: {e}")

    with gr.Blocks(title="DeepSupport Coach 🧑‍🏫", css=MACARON_CSS, theme=theme) as demo:
        gr.HTML(HEADER_HTML)
        with gr.Row(elem_classes=["ds-shell"]):
            with gr.Column(scale=1, elem_classes=["ds-card","ds-card-left"]):
                user_text = gr.Textbox(
                    label="💭 你现在最困扰的一件事是什么？",
                    lines=8,
                    placeholder="直接写就行，不用组织得很完美～",
                    elem_id="ds_user",
                )

                gr.Markdown("#### 🌈 你可能想说 💬", elem_classes=["ds-tip"])
                gr.Examples(
                    examples=[
                        ["我今天汇报被领导骂了，感觉很难受，脑子停不下来。"],
                        ["我最近总是内耗和自我否定，明明没发生大事但就是累。"],
                        ["我跟对象吵架了，一直卡在同一个问题上，不知道怎么沟通。"],
                    ],
                    inputs=[user_text],
                    label="",
                )

                gr.HTML("<div class=\"ds-flex-spacer\"></div>")
                btn = gr.Button("✨ 生成报告", variant="primary", elem_id="ds_btn")
                btn_new = gr.Button("🆕 新对话", variant="primary", elem_id="ds_btn_new")

            with gr.Column(scale=1, elem_classes=["ds-card"]):
                gr.Markdown("### 📝 输出", elem_classes=["ds-out-title"])
                out_md = gr.Markdown(elem_id="ds_out")

        def _run(_user_text: str):
            ut = (_user_text or "").strip()
            if not ut:
                return "请先输入一句话描述：你现在最困扰的一件事是什么？"

            msgs = make_messages(ut, icl_examples=icl_examples, style_injection=style_injection)
            out = generate_one(model, tokenizer, msgs, temperature=float(DEFAULT_TEMP), max_new_tokens=int(DEFAULT_MAX_NEW_TOKENS))
            out = decorate_output(out)

            rec = {
                "time": _now_iso(),
                "uuid": str(uuid.uuid4()),
                "session_id": session_id,
                "base_model": base_model_name,
                "temperature": float(DEFAULT_TEMP),
                "max_new_tokens": int(DEFAULT_MAX_NEW_TOKENS),
                "style_injection": str(style_injection),
                "num_icl": int(len(icl_examples)) if icl_examples else 0,
                "input": ut,
                "output": out,
            }
            _append_jsonl(rec)
            _write_md(rec, out)
            return out

        def _new_chat():
            return "", ""

        btn.click(_run, inputs=[user_text], outputs=[out_md])
        btn_new.click(_new_chat, inputs=None, outputs=[user_text, out_md])

    return demo


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True, help="HF model id or local path (bf16 recommended).")
    p.add_argument("--cache_dir", type=str, default=None, help="HF cache dir.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860, help="Use 0 for auto.")
    p.add_argument("--share", action="store_true", default=False)

    p.add_argument("--use_unsloth", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "bfloat16", "fp16", "float16"])
    p.add_argument("--max_seq_length", type=int, default=4096)

    p.add_argument("--icl_file", type=str, default="", help="JSON/JSONL list of {'user','assistant'} ICL examples (must be sanitized).")
    p.add_argument("--num_icl", type=int, default=2, help="How many ICL examples to inject (0 disables).")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--style_injection", type=str, default="both", choices=["system", "shots", "both"])

    p.add_argument("--save_dir", type=str, default="", help="Directory to save input/output logs (JSONL). Empty disables.")
    p.add_argument("--save_md", action="store_true", help="Also save each interaction as .md under save_dir/md")
    p.add_argument("--session_id", type=str, default="", help="Optional session id for grouping logs")
    return p


def main():
    args = build_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    icl_pool = list(SYNTHETIC_ICL)
    if args.icl_file and str(args.icl_file).strip():
        ext = load_icl_file(args.icl_file)
        print(f"[ICL] loaded external examples: {len(ext)} from {args.icl_file}")
        icl_pool.extend(ext)

    icl_examples = pick_icl(icl_pool, k=int(args.num_icl), seed=int(args.seed)) if int(args.num_icl) > 0 else []
    print(f"[ICL] pool={len(icl_pool)} picked={len(icl_examples)}")

    print(f"Loading model: {args.base_model}")
    model, tokenizer = load_model_tokenizer(
        base_model=args.base_model,
        cache_dir=args.cache_dir,
        use_unsloth=bool(args.use_unsloth),
        dtype=args.dtype,
        max_seq_length=int(args.max_seq_length),
    )

    base_model_name = str(args.base_model)
    session_id = args.session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"

    app = build_app(
        model,
        tokenizer,
        icl_examples=icl_examples,
        style_injection=str(args.style_injection),
        save_dir=str(args.save_dir),
        save_md=bool(args.save_md),
        base_model_name=base_model_name,
        session_id=session_id,
    )

    port = _pick_free_port(int(args.port), host=args.host)
    if port != int(args.port):
        print(f"[Port] {args.port} is busy. Using free port {port} instead.")
    app.launch(server_name=args.host, server_port=port, share=bool(args.share))


if __name__ == "__main__":
    main()



'''


# Example launch command:
CUDA_VISIBLE_DEVICES=0 python scripts/deepsupport_coach_demo.py \
  --base_model Qwen/Qwen2.5-32B-Instruct \
  --icl_file data/icl_synthetic.json \
  --save_dir outputs/coach_logs \
  --save_md \
  --host 0.0.0.0 --port 7865


'''