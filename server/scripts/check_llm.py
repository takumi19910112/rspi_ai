"""Phase 0 の LLM 選定用、再現可能な単発チェックスクリプト.

目的:
    候補 LLM に同じ system prompt（persona.txt）と同じユーザー発話を投げ、
    返答内容とレイテンシをざっくり比較する。会話履歴は持たず、毎回独立リクエスト。
    PR レビューで「私が見たのと同じことを手元で確認できる」状態を作るのが狙い。

使い方:
    # 推奨: ephemeral な httpx を使うので環境を汚さない
    uv run --with httpx --no-project --python 3.12 server/scripts/check_llm.py qwen3.5:4b

    # qwen2.5 など thinking モードを持たないモデルは think フィールド自体を外す
    uv run --with httpx --no-project --python 3.12 server/scripts/check_llm.py qwen2.5:3b --no-think-flag

    # Markdown レポートも保存
    uv run --with httpx --no-project --python 3.12 server/scripts/check_llm.py qwen3.5:4b \\
        --out /tmp/check.md

注意:
    Qwen3.x 系はデフォルトで thinking モード（数分におよぶ内部推論を出力）になり、
    chat エンドポイントが事実上ハングする。`think: false` を送って抑制する必要があり、
    本スクリプトのデフォルト挙動とした。Phase 1 の app/llm.py でも同じフラグを送る前提。
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

# スクリプトの起動ディレクトリに依存させず、リポジトリ root を基準に persona を解決する
REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_PATH = REPO_ROOT / "server" / "app" / "prompts" / "persona.txt"
OLLAMA_URL = "http://localhost:11434/api/chat"

# 子どもらしい質問を 4 種類。事実 / 感情ぶつけ / 知識質問 / 価値観質問 を 1 つずつ入れ、
# モデルの挙動の幅（共感、事実精度、わからない時の素直さ、しつけ寄りの問いへの態度）を見たい
PROMPTS: list[str] = [
    "ねえねえ、そらはどうしてあおいの？",  # 事実質問
    "きょうようちえんで、ともだちとけんかしちゃった。",  # 感情の吐露
    "いちばんつよいどうぶつってなに？",  # 知識 + 主観
    "ぴーまんきらい、たべないとだめ？",  # 価値観 / しつけ
]


def chat_once(
    model: str,
    persona: str,
    user_msg: str,
    *,
    think: bool | None,
    temperature: float,
) -> tuple[str, float]:
    """1 リクエスト = 1 ターン。会話履歴は持たない。返答テキストと実測秒を返す."""
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": persona},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    # think: false は Qwen3.x の reasoning モード抑制。
    # 受け取らないモデルもあるので、--no-think-flag が立っている時は body から落とす
    if think is not None:
        body["think"] = think

    t0 = time.perf_counter()
    r = httpx.post(OLLAMA_URL, json=body, timeout=120.0)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json()["message"]["content"], elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 LLM check (persona + fixed kid prompts)")
    ap.add_argument("model", help="Ollama 上のモデル名。例: qwen3.5:4b")
    ap.add_argument(
        "--no-think-flag",
        action="store_true",
        help="`think` フィールド自体を送らない。Qwen3.x 以外（qwen2.5, llama3.2 等）向け",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=Path, help="Markdown レポートも書き出す（任意）")
    args = ap.parse_args()

    persona = PERSONA_PATH.read_text(encoding="utf-8")
    # --no-think-flag が立っていればフィールド省略、無ければ false を送って thinking 抑制
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
