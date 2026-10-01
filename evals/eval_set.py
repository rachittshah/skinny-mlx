"""Decode-speed eval set mirroring a local chat product's traffic (inference-lab workloads).

Categories follow the lab's production surfaces: RAG-grounded answers, explanatory chat, code
generation / refactoring, agent tool-call JSON, summarization, and step-by-step reasoning.
Hand-written (blog guidance: 5-10 manual cases per surface before any synthesis), each sized to
elicit a few hundred tokens so decode speed — not prefill — dominates. Stratified split: every
category appears in both train (hill-climb reads these) and test (held out, only scored).

RAG context is excerpted from this repo's own public docs so nothing private is embedded.
"""

from pathlib import Path

_DOCS = Path(__file__).resolve().parent.parent / "docs"


def _excerpt(name: str, start: str, n_chars: int = 1800) -> str:
    text = (_DOCS / name).read_text()
    i = text.find(start)
    return text[i : i + n_chars] if i >= 0 else text[:n_chars]


def _rag(question: str, doc: str, anchor: str) -> str:
    return ("Answer the question using only the context. Cite the relevant numbers.\n\n"
            f"<context>\n{_excerpt(doc, anchor)}\n</context>\n\nQuestion: {question}")


_TOOLS = """You can call tools by replying with JSON: {"tool": name, "args": {...}}.
Tools: search_docs(query: str, k: int), calculator(expression: str), get_time(tz: str), create_ticket(title: str, body: str, priority: "low"|"med"|"high").
Plan step by step, then output each tool call you would make as a JSON object on its own line, then a final answer."""

PROMPTS = [
    # --- RAG-grounded answers
    dict(id="rag-roofline", cat="rag", split="train",
         text=_rag("Why can't a faster kernel alone deliver a 4x decode speedup on this Mac? Explain with the measured numbers.",
                   "roofline-and-plan.md", "## 2. Baselines vs roofline")),
    dict(id="rag-deadzone", cat="rag", split="train",
         text=_rag("What is the skinny-GEMM dead zone and what does it do to speculative decoding? Walk through the table.",
                   "roofline-and-plan.md", "## 3. The real bottleneck")),
    dict(id="rag-hybrid", cat="rag", split="test",
         text=_rag("Summarize the approaches to speculative decoding on hybrid recurrent models and compare their trade-offs.",
                   "lit-2026.md", "## Speculation on hybrid")),
    # --- explanatory chat
    dict(id="chat-cache", cat="chat", split="train",
         text="Explain how a CPU cache hierarchy works, including L1, L2, L3, cache lines, and coherence protocols like MESI."),
    dict(id="chat-tcp", cat="chat", split="train",
         text="Explain what happens, step by step, when I type a URL into a browser and press enter, from DNS to rendering."),
    dict(id="chat-gc", cat="chat", split="test",
         text="Compare tracing garbage collection and reference counting. Cover pauses, cycles, throughput, and give examples of languages using each."),
    # --- code
    dict(id="code-refactor", cat="code", split="train",
         text="Rewrite this Python class adding type hints and docstrings, keep all logic:\n\nclass Stack:\n    def __init__(self):\n"
              "        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n"
              "        if not self.items:\n            raise IndexError('pop from empty stack')\n        return self.items.pop()\n"
              "    def peek(self):\n        if not self.items:\n            raise IndexError('peek from empty stack')\n"
              "        return self.items[-1]\n    def __len__(self):\n        return len(self.items)\n"),
    dict(id="code-lru", cat="code", split="train",
         text="Write a Python LRU cache class with get and put in O(1) using a dict and a doubly linked list, with type hints, docstrings, and a short usage example."),
    dict(id="code-sql", cat="code", split="test",
         text="Write a FastAPI endpoint that paginates a list of users from SQLite using parameterized queries, with a Pydantic response model and error handling."),
    # --- agent / tool JSON
    dict(id="tool-incident", cat="tool", split="train",
         text=f"{_TOOLS}\n\nTask: Our p95 TTFT doubled since yesterday. Find the relevant docs, compute the percentage increase from 0.44s to 0.89s, and open a ticket."),
    dict(id="tool-planning", cat="tool", split="test",
         text=f"{_TOOLS}\n\nTask: Schedule a load test at 9am Tokyo time, look up our documented concurrency limits, compute 16 users x 200 tokens, and file a low priority ticket summarizing the plan."),
    # --- summarization
    dict(id="sum-design", cat="summarize", split="train",
         text="Summarize the following design note in 8 bullet points for an engineering manager:\n\n"
              + _excerpt("flatspec-design.md", "# FlatSpec", 2200)),
    dict(id="sum-lit", cat="summarize", split="test",
         text="Summarize the following literature notes as a short report with sections Problem, Approaches, Open gaps:\n\n"
              + _excerpt("lit-2026.md", "## The consensus problem", 2200)),
    # --- step-by-step reasoning
    dict(id="math-train", cat="reason", split="train",
         text="A train leaves at 9:40 at 72 km/h; another leaves the same station at 10:05 at 90 km/h on the same track. "
              "When and where does the second catch the first? Reason step by step and box the final answer."),
    dict(id="math-fn", cat="reason", split="test",
         text="The function f satisfies f(x) + f(y) = f(x + y) - xy - 1 for all real x, y. If f(1) = 1, find all integers n "
              "such that f(n) = n. Reason step by step and box the final answer."),
    dict(id="math-prob", cat="reason", split="train",
         text="Two dice are rolled. What is the probability the sum is prime, given at least one die shows an odd number? "
              "Reason step by step and box the final answer."),
]


def split(name: str) -> list[dict]:
    return [p for p in PROMPTS if p["split"] == name]
