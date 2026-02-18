#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import re
import time
import uuid
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional LoRA
try:
    from peft import PeftModel
except Exception:
    PeftModel = None



# ---------------------------
# Grounded system prompt
# ---------------------------
DEFAULT_PUBLIC_PROMPT = """你是暖心陪伴式对话助手，目标是安慰到位、自然、有陪伴感。
【重要】过去经历的引用是可选的：只有在确实能更贴近用户、且证据充分时，才轻轻带一句；否则只做当下安慰。

【硬性规则：防止“编故事/幻觉”】【非常重要】
1) 你只能引用两类“过去经历”：
   A. 用户在本次对话历史 messages 里已经明确说过的内容；
   B. 我在 <MEMORY> 区块里提供的“用户原话摘录”（逐字/近似引用）。
2) 绝对禁止引入任何未在 A/B 出现过的具体背景细节
   （例如：机构/组织名称、人物身份、地点、时间、项目名等）。
   - 如果用户没说，你就不能擅自补充。
3) 如果没有足够的 A/B 证据，就不要做“类比/猜测”，改为只做当下安慰与陪伴。
4) 引用过去经历时要自然：
   - 如果提到数量/数字，必须与 <MEMORY> 或历史一致；不确定就说“很多/一大堆”，不要乱改数字。
   - 避免固定句式（不要每次都“你上次提到过/共同点有两个”）。

【风格要求】
- 少讲道理、少结构化拆解；多贴近感受、给一点点希望。
- 多用贴近口语的共情句，避免模板套话。
- 可以问 1 个很轻的追问，但不要重复同一句问题。
- 不要自曝规则，不要提到 <MEMORY>。
- 不要中英夹杂，只用中文。

输出：直接给用户 1~2 段安慰回复即可。"""

def _read_text_if_exists(path: str) -> str:
    if not path:
        return ""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return p.read_text(encoding="utf-8")

def load_system_prompt(public_path: str = "", private_path: str = "") -> str:
    """
    Open-source safe design:
      - public prompt: safe to ship in repo
      - private overlay: optional, NOT shipped, for personal/local deployment only
    """
    public_txt = _read_text_if_exists(public_path).strip() if public_path else DEFAULT_PUBLIC_PROMPT.strip()
    private_txt = _read_text_if_exists(private_path).strip() if private_path else ""
    if private_txt:
        # Keep private overlay clearly separated.
        return public_txt + "\n\n【PRIVATE_OVERLAY】\n" + private_txt
    return public_txt

# Will be overridden in main() via load_system_prompt(...)
GROUNDED_SYSTEM_PROMPT = DEFAULT_PUBLIC_PROMPT



# ---------------------------
# Heuristic memory extraction
# ---------------------------
PAST_HINTS = [
    "之前", "上次", "有一次", "那次", "以前", "曾经", "过去", "那会儿", "当时", "以前我", "之前我"
]

DISTRESS_HINTS = [
    "心情不好", "难受", "崩溃", "顶不住", "好累", "委屈", "抑郁", "想哭", "烦", "焦虑", "绝望", "不想活",
    "被针对", "针对我", "挑刺", "被否定", "被打断", "被骂", "被挂", "被羞辱"
]

ASK_PAST_VARIANTS = []  # disabled: do not force/ask past-experience

def _contains_any(text: str, hints: List[str]) -> bool:
    t = text or ""
    return any(h in t for h in hints)

def _extract_quote(text: str) -> Optional[str]:
    """Extract a concise 'user quote' that looks like a past-experience snippet."""
    if not text:
        return None
    t = text.strip()
    # Keep max length to avoid overlong memory
    t = re.sub(r"\s+", " ", t)
    if len(t) < 6:
        return None
    # Only store when user explicitly signals past (avoid '有吧' etc.)
    if _contains_any(t, PAST_HINTS):
        # take first sentence-ish chunk
        m = re.split(r"[。！？!?\n]", t)
        q = m[0].strip()
        return q[:160] if q else None
    return None

def _dedup_append(mem: List[str], quote: str, max_keep: int = 12) -> List[str]:
    quote = quote.strip()
    if not quote:
        return mem
    # simple dedup by substring similarity
    for x in mem:
        if quote == x:
            return mem
        if quote in x or x in quote:
            return mem
    mem.append(quote)
    if len(mem) > max_keep:
        mem = mem[-max_keep:]
    return mem


# ---------------------------
# Minimal safety / hallucination cleanup
# ---------------------------
def _context_contains_word(history_pairs: List[Tuple[str, str]], mem_quotes: List[str], word: str) -> bool:
    for u, b in history_pairs:
        if word in (u or "") or word in (b or ""):
            return True
    for q in mem_quotes:
        if word in q:
            return True
    return False

def _light_dehallucinate(text: str, history_pairs: List[Tuple[str, str]], mem_quotes: List[str]) -> str:
    """Open-source minimal postprocess (no scenario/institution keyword lists)."""
    out = (text or "").strip()
    # Avoid making incorrect time claims
    out = out.replace("你上次提到过", "你提到过")
    out = out.replace("你刚刚说到", "你提到过")
    out = re.sub(r"[ 	]+", " ", out).strip()
    return out

def _fix_number_paraphrase(text: str, history_pairs: List[Tuple[str, str]], mem_quotes: List[str]) -> str:
    """Open-source minimal cleanup (no special-case numeric rules)."""
    out = (text or "").strip()

    # Clean a few awkward phrase patterns that occasionally appear
    out = out.replace("你之前有一也是", "你之前也有一次")
    out = out.replace("你之前有一也是做了", "你之前也有一次做了")
    out = re.sub(r"你之前有一(次)?也是", "你之前也有一次", out)

    # Keep time-neutral wording
    out = out.replace("你上次提到过", "你提到过")
    out = out.replace("你刚刚说到", "你提到过")
    out = out.replace("共同点有两个", "有几点相似")

    out = re.sub(r"[ 	]+", " ", out).strip()
    return out



# ---------------------------
# Persistence: save turns + optional memory load
# ---------------------------
def _ensure_parent_dir(path: str):
    if not path:
        return
    p = pathlib.Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)

def _load_memory_from_json(log_path: str, max_pairs: int = 8) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Load last N turns from a JSON array file written by this script.
    Returns (history_pairs, mem_quotes_from_last_turn).
    """
    if not log_path:
        return [], []
    p = pathlib.Path(log_path)
    if not p.exists():
        return [], []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return [], []
        tail = data[-max_pairs:]
        pairs: List[Tuple[str, str]] = []
        for r in tail:
            if not isinstance(r, dict):
                continue
            u = (r.get("user") or "").strip()
            b = (r.get("assistant") or "").strip()
            if u and b:
                pairs.append((u, b))
        last_mem: List[str] = []
        for r in reversed(data):
            if isinstance(r, dict) and isinstance(r.get("mem_quotes"), list):
                last_mem = [str(x) for x in r["mem_quotes"] if str(x).strip()]
                break
        return pairs, last_mem
    except Exception:
        return [], []

def _append_turn_to_json(log_path: str, record: Dict):
    """Append one turn to a JSON array file (read-modify-write).
    Also writes a .jsonl sidecar for streaming/debug.
    """
    if not log_path:
        return
    _ensure_parent_dir(log_path)
    p = pathlib.Path(log_path)

    # Load existing array
    arr = []
    if p.exists():
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    arr.append(record)
    p.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

    # JSONL sidecar
    p_jsonl = p.with_suffix(p.suffix + "l") if not str(p).endswith(".jsonl") else p
    try:
        with open(p_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------------------
# Model wrapper
# ---------------------------
@dataclass
class ChatState:
    history: List[Tuple[str, str]]
    mem_quotes: List[str]
    last_ask_id: str
    last_user_quote: str
    session_id: str  # for logging
    memory_cut: int = 0  # history[:memory_cut] treated as loaded memory; history[memory_cut:] is live chat

def _new_state(session_id: str = "") -> ChatState:
    sid = session_id or str(uuid.uuid4())
    return ChatState(history=[], mem_quotes=[], last_ask_id="", last_user_quote="", session_id=sid, memory_cut=0)

def load_model(model_name: str, lora_path: Optional[str], dtype: str, attn_implementation: str):
    dt = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(lora_path or model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        dtype=dt,
        device_map="auto",
    )
    if lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but --lora was provided. `pip install peft`")
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        model = base
    model.eval()
    return tok, model


def _token_len(tok, messages: List[Dict[str, str]]) -> int:
    """Approx token length of the full prompt (chat template)."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tok(prompt, add_special_tokens=False).input_ids)

def _build_messages(tok, state: ChatState, user_text: str,
                   enable_memory: bool = True,
                   max_context_tokens: int = 8192,
                   max_new_tokens: int = 256) -> List[Dict[str, str]]:
    """Dynamic history packing:
    - Always prioritize LIVE multi-turn chat (history[memory_cut:])
    - Only add loaded memory (history[:memory_cut]) if budget allows
    - Keeps chronological order
    """
    # Build MEMORY block from user quotes (evidence only)
    if state.mem_quotes:
        mem_block = "\n".join([f"- {q}" for q in state.mem_quotes[-4:]])
    else:
        mem_block = "(empty)"

    base: List[Dict[str, str]] = []
    base.append({"role": "system", "content": GROUNDED_SYSTEM_PROMPT})
    base.append({"role": "system", "content": f"<MEMORY>\n{mem_block}\n</MEMORY>"})
    if state.last_user_quote:
        base.append({"role": "system", "content": f"<EVIDENCE_LAST_QUOTE>\n- {state.last_user_quote}\n</EVIDENCE_LAST_QUOTE>"})

    # Budget: leave room for generation + a small safety margin
    budget = max(256, int(max_context_tokens) - int(max_new_tokens) - 64)

    # Split history into memory vs live
    mem_pairs = state.history[: max(0, int(state.memory_cut))]
    live_pairs = state.history[max(0, int(state.memory_cut)) :]

    # Helper: pick as many pairs from the tail as fit, preserving order
    def pick_tail(pairs):
        chosen_rev = []
        for u, b in reversed(pairs):
            cand = list(reversed(chosen_rev))
            # add this pair at the beginning of cand (because we're going backwards)
            cand = [(u, b)] + cand
            msgs = base.copy()
            for uu, bb in cand:
                msgs.append({"role": "user", "content": uu})
                msgs.append({"role": "assistant", "content": bb})
            msgs.append({"role": "user", "content": user_text})
            if _token_len(tok, msgs) <= budget:
                chosen_rev.append((u, b))
            else:
                break
        return list(reversed(chosen_rev))

    # 1) Always pack LIVE chat first
    live_keep = pick_tail(live_pairs)

    # 2) If enabled, then pack MEMORY if still within budget
    mem_keep: List[Tuple[str, str]] = []
    if enable_memory and mem_pairs:
        # compute remaining budget by checking current prompt length
        msgs_now = base.copy()
        for uu, bb in live_keep:
            msgs_now.append({"role": "user", "content": uu})
            msgs_now.append({"role": "assistant", "content": bb})
        msgs_now.append({"role": "user", "content": user_text})
        used = _token_len(tok, msgs_now)
        remaining = budget - used
        if remaining > 0:
            # pack memory from tail, but need to consider combined length
            chosen_rev = []
            for u, b in reversed(mem_pairs):
                cand_mem = list(reversed(chosen_rev))
                cand_mem = [(u, b)] + cand_mem
                msgs = base.copy()
                for uu, bb in cand_mem + live_keep:
                    msgs.append({"role": "user", "content": uu})
                    msgs.append({"role": "assistant", "content": bb})
                msgs.append({"role": "user", "content": user_text})
                if _token_len(tok, msgs) <= budget:
                    chosen_rev.append((u, b))
                else:
                    break
            mem_keep = list(reversed(chosen_rev))

    messages: List[Dict[str, str]] = base.copy()
    for u, b in mem_keep + live_keep:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_text})
    return messages



def _generate(tok, model, messages, max_new_tokens=256):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def warm_step(tok, model, user_text: str, state: ChatState, max_new_tokens: int = 256,
              save_json: str = "", enable_memory: bool = True, max_context_tokens: int = 8192) -> Tuple[List[Tuple[str, str]], ChatState]:
    user_text = (user_text or "").strip()
    if not user_text:
        return state.history, state

    # Update memory quotes from user's current input (only user's own text)
    q = _extract_quote(user_text)
    if q:
        state.mem_quotes = _dedup_append(state.mem_quotes, q)
        state.last_user_quote = q
    # Do NOT force ask past experience. Always prioritize present-moment support.

    # Otherwise, call model
    messages = _build_messages(tok, state, user_text, enable_memory=enable_memory, max_context_tokens=max_context_tokens, max_new_tokens=max_new_tokens)
    bot = _generate(tok, model, messages, max_new_tokens=max_new_tokens)

    # light de-hallucination for known issue words
    bot = _light_dehallucinate(bot, state.history, state.mem_quotes)
    bot = _fix_number_paraphrase(bot, state.history, state.mem_quotes)

    # Append to history
    state.history.append((user_text, bot))

    # Save to JSON (one record per user->assistant turn)
    if save_json:
        rec = {
            "ts": time.time(),
            "session_id": state.session_id,
            "user": user_text,
            "assistant": bot,
            "mem_quotes": state.mem_quotes,
        }
        _append_turn_to_json(save_json, rec)

    return state.history, state


# ---------------------------
# UI: Gradio + optional CLI
# ---------------------------

# Macaron UI (match the "Coach" demo vibe, but keep vertical chat layout)
MACARON_CSS = r'''
:root{
  --ds-ink:#2b2f55;
  --ds-ink-soft: rgba(43,47,85,.72);
  --ds-card: rgba(255,255,255,.68);
  --ds-card-border: rgba(47,47,70,.10);
  --ds-shadow: 0 24px 60px rgba(47,47,70,.14);
  --ds-radius: 26px;

  --ds-accent:  rgba(255,111,177,1);
  --ds-accent2: rgba(109,232,255,1);
  --ds-accent3: rgba(125,255,176,1);
  --ds-accent4: rgba(196,155,255,1);
  --ds-accent5: rgba(255,240,122,1);
  --ds-accent6: rgba(255,180,120,1);
  --ds-accent7: rgba(160,170,255,1);
  --ds-basew: 980px;
  --ds-widew: 1120px;
  --ds-shiftX: -18px;
}

.gradio-container{
  min-height: 100vh;
  padding-bottom: 14px;
}

/* soft floating blobs */
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

/* header */
#ds_header{
  position: relative;
  z-index: 1;
  max-width: 980px;
  margin: 10px auto 6px auto;
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
  line-height: 1.5;
  font-size: 14px;
  color: var(--ds-ink-soft);
}

/* shell + card */
.ds-shell{
  position: relative;
  z-index: 1;
  max-width: 980px;
  margin: 0 auto;
  padding: 0px 10px 0 10px;
}

/* widen the whole frame to the RIGHT on large screens (keep left edge the same) */
@media (min-width: 1260px){
  /* Wider layout but keep whole block visually centered; shift slightly left if desired */
  #ds_header,
  .ds-shell{
    max-width: var(--ds-widew);
    margin-left: calc((100% - var(--ds-widew))/2 + var(--ds-shiftX));
    margin-right: auto;
  }
}

.ds-card{
  background: var(--ds-card);
  border: 1px solid var(--ds-card-border);
  border-radius: var(--ds-radius);
  box-shadow: var(--ds-shadow);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 6px;
}

/* chat wrapper */
#ds_chat_wrap{ position: relative; }
#ds_chatbot{
  border-radius: 22px !important;
  overflow: hidden;
  border: 1px solid rgba(47,47,70,.10);
  background: rgba(255,255,255,.55);
}

/* overlay chips inside chat area */
#ds_overlay{
  position: relative;
  z-index: 2;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  margin: 0;
}
#ds_overlay .gr-button{
  margin: 0 !important;
}

/* suggestion box (make background truly transparent; keep chip shadows) */
#ds_suggest_box{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 4px 0 !important;
  border-radius: 0 !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

/* Gradio often adds a grey panel background on internal wrappers.
   Nuke wrapper backgrounds ONLY inside the suggest area (do not touch inputs/checkbox). */
#ds_suggest_box .wrap,
#ds_suggest_box .gr-group,
#ds_suggest_box .gr-form,
#ds_suggest_box .gr-panel,
#ds_suggest_box .gr-box,
#ds_suggest_box .container,
#ds_suggest_box div{
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}
/* Title: plain, not bold */
#ds_suggest_title{
  margin: 0 0 6px 2px !important;
}
#ds_suggest_title,
#ds_suggest_title *{
  font-weight: 400 !important;
  color: var(--ds-ink) !important;
}

/* Chips layout */
#ds_overlay{
  gap: 8px;
  flex-wrap: wrap;
}
#ds_memrow{
  margin-top: 4px;
  display:flex;
  align-items:center;
}
.ds-chip,
.ds-chip button,
button.ds-chip{
  /* Suggested-question chips: match Coach UI surface */
  white-space: nowrap !important;
  border-radius: 999px !important;

  /* Coach uses a slightly stronger translucent white card */
  background: rgba(255,255,255,.80) !important;
  border: 1px solid rgba(255,255,255,.58) !important;

  /* Keep the depth + pressed feel */
  box-shadow: 0 10px 22px rgba(47,47,70,.10) !important;

  font-weight: 400 !important;
  color: var(--ds-ink) !important;

  padding: 8px 14px !important;
}
.ds-chip:hover,
.ds-chip:hover button,
button.ds-chip:hover{
  background: rgba(255,255,255,.90) !important;
  box-shadow: 0 14px 28px rgba(47,47,70,.14) !important;
}


.ds-chip,
.ds-chip button,
button.ds-chip{
  /* Suggested-question chips: match Coach UI surface */
  white-space: nowrap !important;
  border-radius: 999px !important;

  /* Coach uses a slightly stronger translucent white card */
  background: rgba(255,255,255,.80) !important;
  border: 1px solid rgba(255,255,255,.58) !important;

  /* Keep the depth + pressed feel */
  box-shadow: 0 10px 22px rgba(47,47,70,.10) !important;

  font-weight: 400 !important;
  color: var(--ds-ink) !important;

  padding: 8px 14px !important;
}
.ds-chip:hover,
.ds-chip:hover button,
button.ds-chip:hover{
  background: rgba(255,255,255,.90) !important;
  box-shadow: 0 14px 28px rgba(47,47,70,.14) !important;
}



/* Keep subtle shadow on chips (including press/active) */
.ds-chip button{
  box-shadow: 0 10px 22px rgba(47,47,70,.06) !important;
  transition: transform .08s ease, filter .08s ease, box-shadow .08s ease;
}
.ds-chip button:hover{
  filter: brightness(1.02);
  box-shadow: 0 12px 26px rgba(47,47,70,.10) !important;
}
.ds-chip button:active{
  transform: translateY(0px) scale(.99);
  box-shadow: 0 8px 18px rgba(47,47,70,.10) !important;
}
.ds-chip{ background: transparent !important; border: none !important; box-shadow: none !important; }
#ds_memcheck{
  margin-left: auto;
}
#ds_memcheck label{
  font-weight: 800 !important;
  color: var(--ds-ink) !important;
  font-size: 13px !important;
}
input[type="checkbox"]{ accent-color: var(--ds-accent) !important; }

/* input */
#ds_user textarea,
#ds_user input{
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
#ds_btn_send button{
  width: 100%;
  border-radius: 999px !important;
  border: none !important;
  background: linear-gradient(90deg,var(--ds-accent),var(--ds-accent2)) !important;
  box-shadow: 0 14px 34px rgba(255,143,184,.24);
  font-weight: 900 !important;
  letter-spacing: .4px;
  transition: transform .08s ease, filter .08s ease;
}
#ds_btn_send button:hover{ transform: translateY(-1px); filter: brightness(1.02); }
#ds_btn_send button:active{ transform: translateY(0px) scale(.99); }



#ds_btn_clear button{
  width: 100%;
  border-radius: 999px !important;
  border: none !important;
  background: linear-gradient(90deg,var(--ds-accent),var(--ds-accent2)) !important;
  box-shadow: 0 14px 34px rgba(255,143,184,.24);
  font-weight: 900 !important;
  letter-spacing: .4px;
  transition: transform .08s ease, filter .08s ease;
}
#ds_btn_clear button:hover{ transform: translateY(-1px); filter: brightness(1.02); }
#ds_btn_clear button:active{ transform: translateY(0px) scale(.99); }

/* small helper text */
.ds-tip{
  color: var(--ds-ink-soft) !important;
  font-size: 12px !important;
  margin-top: 4px;
}

/* prevent focus/active outlines from creating square highlight blocks */
.ds-chip button:focus,
.ds-chip button:focus-visible{
  outline: none !important;
  box-shadow: 0 10px 22px rgba(47,47,70,.06) !important;
}
'''

HEADER_HTML = r'''
<div id="ds_header">
  <div class="ds-title">✨ <span class="grad">DeepSupport Warm ❤️‍🩹</span> 🫧</div>
  <div class="ds-sub">先把你的情绪接住，再一起往前走 🫂</div>
</div>
'''

def build_ui(tok, model, args):
    blocks_kwargs = dict(title="DeepSupport Warm ❤️‍🩹", css=MACARON_CSS)
    try:
        blocks_kwargs["theme"] = gr.themes.Soft()
    except Exception:
        try:
            blocks_kwargs["theme"] = gr.themes.Default()
        except Exception:
            pass

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(HEADER_HTML)

        # Initialize state (optionally load memory from saved JSON)
        if args.enable_memory and args.save_json:
            hist0, mem0 = _load_memory_from_json(args.save_json, max_pairs=args.memory_k)
            s0 = _new_state()
            s0.history = hist0
            s0.mem_quotes = mem0
            s0.memory_cut = len(hist0)
            st = gr.State(asdict(s0))
        else:
            st = gr.State(_new_state())

        with gr.Row(elem_classes=["ds-shell"]):
            with gr.Column(elem_classes=["ds-card"], elem_id="ds_chat_wrap"):
                chatbot = gr.Chatbot(label="", height=320, elem_id="ds_chatbot")  # tuples mode for compatibility


                # keep log path but hide it from UI
                log_path = gr.Textbox(value=args.save_json, label="保存文件（JSON）", interactive=False, visible=False)

                user = gr.Textbox(
                    label="💭 说说你现在的感受",
                    placeholder="直接写就行，不用组织得很完美～",
                    lines=1,
                    elem_id="ds_user",
                )

                # Suggested questions + memory toggle (placed between input and send button)
                with gr.Group(elem_id="ds_suggest_box"):
                    gr.Markdown("你可能想说 💬", elem_id="ds_suggest_title")
                    with gr.Row(elem_id="ds_overlay"):
                        chip1 = gr.Button("😮‍💨 我今天有点撑不住了…", elem_classes=["ds-chip"], variant="secondary")
                        chip2 = gr.Button("😣 我一直在内耗，停不下来。", elem_classes=["ds-chip"], variant="secondary")
                        chip3 = gr.Button("🥺 我需要被理解，但不知道怎么说。", elem_classes=["ds-chip"], variant="secondary")

                    with gr.Row(elem_id="ds_memrow"):
                        use_mem = gr.Checkbox(
                            value=bool(args.enable_memory),
                            label="🧠 启用历史记忆",
                            elem_id="ds_memcheck",
                        )

                with gr.Row():
                    btn = gr.Button("📨 发送", variant="primary", elem_id="ds_btn_send")
                    btn_clear = gr.Button("💬 新对话", variant="primary", elem_id="ds_btn_clear")

        # --- interactions ---
        def _chat(user_text, chat, state_dict, use_mem_flag, log_path_value):
            if isinstance(state_dict, dict):
                if "last_user_quote" not in state_dict:
                    state_dict["last_user_quote"] = ""
                state_dict.setdefault("session_id", str(uuid.uuid4()))
                state = ChatState(**state_dict)
            else:
                state = state_dict

            ut = (user_text or "").strip()
            if ut.lower() == "/reset":
                sid = getattr(state, "session_id", "")
                state = _new_state(sid)
                return "", [], asdict(state)

            chat, state = warm_step(
                tok, model, ut, state,
                max_new_tokens=args.max_new_tokens,
                save_json=(log_path_value or args.save_json),
                enable_memory=bool(use_mem_flag),
                max_context_tokens=args.max_context_tokens,
            )
            return "", chat, asdict(state)

        # chips: fill textbox (no queue to feel instant)
        chip1.click(lambda: "我今天有点撑不住了…", outputs=[user], queue=False)
        chip2.click(lambda: "我一直在内耗，停不下来。", outputs=[user], queue=False)
        chip3.click(lambda: "我需要被理解，但不知道怎么说。", outputs=[user], queue=False)

        btn.click(_chat, inputs=[user, chatbot, st, use_mem, log_path], outputs=[user, chatbot, st])
        user.submit(_chat, inputs=[user, chatbot, st, use_mem, log_path], outputs=[user, chatbot, st])

        def _clear():
            state = _new_state()
            return [], asdict(state)

        btn_clear.click(_clear, outputs=[chatbot, st])

    return demo
def cli_loop(tok, model, args):
    if args.enable_memory and args.save_json:
        hist0, mem0 = _load_memory_from_json(args.save_json, max_pairs=args.memory_k)
        state = _new_state()
        state.history = hist0
        state.mem_quotes = mem0
        state.memory_cut = len(hist0)
    else:
        state = _new_state()
    print("\nDeepSupport Warm CLI (Grounded)  commands: /reset, /quit\n")
    while True:
        try:
            u = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not u:
            continue
        if u.lower() in ["/quit", "quit", "exit"]:
            print("Bye.")
            break
        if u.lower() == "/reset":
            state = _new_state()
            print("System> 已开启新对话。")
            continue
        _, state = warm_step(tok, model, u, state, max_new_tokens=args.max_new_tokens, save_json=args.save_json,
                               enable_memory=bool(args.enable_memory), max_context_tokens=args.max_context_tokens)
        print(f"Bot> {state.history[-1][1]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base HF model, e.g. Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--lora", default="", help="Optional LoRA dir, e.g. outputs/warm_lora")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn_implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3", "flex_attention"])
    ap.add_argument("--prompt_public", default="",
                    help="Path to a public system prompt .txt. If empty, uses the built-in DEFAULT_PUBLIC_PROMPT.")
    ap.add_argument("--prompt_private", default="",
                    help="Optional path to a private overlay prompt .txt (NOT shipped in open-source).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_context_tokens", type=int, default=8192,
                    help="Max input context tokens budget (live chat prioritized; memory added only if budget allows).")
    ap.add_argument("--save_json", default="",
                    help="Optional. If set, save each user->assistant turn to this JSON array file (also writes a .jsonl sidecar).")
    ap.add_argument("--enable_memory", action="store_true",
                help="If set, load previous turns from --save_json and use them as context.")
    ap.add_argument("--memory_k", type=int, default=8,
                help="How many previous turns to load/use as memory context when --enable_memory is set.")
    ap.add_argument("--cli", action="store_true", help="Run terminal chat instead of Gradio")
    args = ap.parse_args()
    # Load (public + optional private) system prompt
    global GROUNDED_SYSTEM_PROMPT
    GROUNDED_SYSTEM_PROMPT = load_system_prompt(args.prompt_public, args.prompt_private)


    lora_path = args.lora.strip() or None
    tok, model = load_model(args.model, lora_path, args.dtype, args.attn_implementation)

    if args.cli:
        cli_loop(tok, model, args)
    else:
        demo = build_ui(tok, model, args)
        demo.queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()


'''

module load mamba
conda activate /condo/wanglab/shared/conda_envs/qwen

cd /home/tmhyxc55/wanglab/tmhyxc55/2d-3d/my/deepsupport-oss-opensource/DeepSupport_Warm

export HF_HOME=/home/tmhyxc55/wanglab/tmhyxc55/.cache/huggingface
export HF_HUB_CACHE=/home/tmhyxc55/wanglab/tmhyxc55/.cache/huggingface/hub

CUDA_VISIBLE_DEVICES=5 python scripts/deepsupport_warm_demo.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --lora Yukyin/deepsupport-warm-lora-oss \
  --save_json outputs/warm_chat_history.json \
  --enable_memory \
  --memory_k 12 \
  --host 0.0.0.0 --port 7866

'''