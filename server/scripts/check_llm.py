"""Provisional LLM check: send persona + fixed prompts to Ollama, print replies and timings.

Run from anywhere; paths resolve relative to the repo root.

Examples:
    uv run --with httpx python server/scripts/check_llm.py qwen3.5:4b
    uv run --with httpx python server/scripts/check_llm.py qwen2.5:3b --no-think-flag
    uv run --with httpx python server/scripts/check_llm.py qwen3.5:4b --out server/scripts/results/check_qwen3.5-4b.md

Qwen3.x models default to thinking mode (long internal reasoning). We send
`think=false` to disable it. Pass --no-think-flag for models that reject the field.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_PATH = REPO_ROOT / "server" / "app" / "prompts" / "persona.txt"
OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPTS: list[str] = [
    "ねえねえ、そらはどうしてあおいの？",
    "きょうようちえんで、ともだちとけんかしちゃった。",
    "いちばんつよいどうぶつってなに？",
    "ぴーまんきらい、たべないとだめ？",
]


def chat_once(
    model: str,
    persona: str,
    user_msg: str,
    *,
    think: bool | None,
    temperature: float,
) -> tuple[str, float]:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": persona},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if think is not None:
        body["think"] = think
    t0 = time.perf_counter()
    r = httpx.post(OLLAMA_URL, json=body, timeout=120.0)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json()["message"]["content"], elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="Ollama model name, e.g. qwen3.5:4b")
    ap.add_argument(
        "--no-think-flag",
        action="store_true",
        help="Omit the `think` field (use for models that reject it)",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=Path, help="Also write Markdown report to this path")
    args = ap.parse_args()

    persona = PERSONA_PATH.read_text(encoding="utf-8")
    think: bool | None = None if args.no_think_flag else False

    header_lines = [
        f"# LLM check — `{args.model}`",
        "",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- temperature: {args.temperature}",
        f"- think field: {'omitted' if think is None else think}",
        f"- persona: `{PERSONA_PATH.relative_to(REPO_ROOT)}`",
        "",
    ]
    md_lines: list[str] = list(header_lines)
    print("\n".join(header_lines))

    for q in PROMPTS:
        reply, sec = chat_once(args.model, persona, q, think=think, temperature=args.temperature)
        block = [
            f"## Q: {q}",
            "",
            f"**A** ({sec * 1000:.0f} ms):",
            "",
            reply.strip(),
            "",
        ]
        print("\n".join(block))
        md_lines.extend(block)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
