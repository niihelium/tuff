#!/usr/bin/env python3
"""Generate G9 tokenizer fixtures from llama-server /tokenize."""
import argparse, json, sys, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--out", type=Path, default=HERE / "fixtures/g9_tokenizer.json")
    a = ap.parse_args()
    corpus = json.loads((ROOT / "prompts_corpus.json").read_text())
    texts = [("corpus_%d" % i, c["prompt"]) for i, c in enumerate(corpus)]
    texts += [
        ("english", "The quick brown fox jumps over the lazy dog."),
        ("chinese", " Transcendence 是一个项目。今天天气很好。"),
        ("japanese", "東京は日本の首都です。ラーメンが好きです。"),
        ("korean", "안녕하세요. 반갑습니다."),
        ("arabic", "مرحبا بالعالم، هذا اختبار."),
        ("russian", "Привет, мир! Это тест токенизатора."),
        ("hindi", "नमस्ते दुनिया, यह एक परीक्षण है।"),
        ("emoji", "🚀🔥🧪👩‍💻🇳🇴 — emoji and flags."),
        ("code_c", "int main(void) { return 0; } // comment\n"),
        ("code_python", "def f(x):\n    return {'a': [1, 2, 3]}\n"),
        ("code_json", '{"a":1,"b":[true,null,2.5],"s":"x\\ny"}'),
        ("whitespace", "  \t\n  leading and trailing  \n\n"),
        ("special", "literal <|im_start|>user<|im_end|> <think> <tool_call>"),
        ("numbers", "1234567890 3.14159 -42 1e-9 0xdeadbeef"),
        ("mixed", "Hello 世界 🌍 — ¿qué tal? Ça va?"),
        ("empty", ""),
        ("long", "lorem ipsum " * 300),
    ]
    out = []
    for name, text in texts:
        tok = post(a.port, "/tokenize", {"content": text})["tokens"]
        out.append({"name": name, "text": text, "tokens": tok})
        print(f"{name}: {len(tok)} tokens", flush=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {a.out} ({len(out)} cases)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
