# Resume Depth

Upload a resume in the browser and it is scored on four dimensions. The model is
pluggable — run it locally for privacy, or on Groq for speed.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env             # then fill in GROQ_API_KEY
python -m uvicorn app:app --port 8000
```

Open <http://localhost:8000>. The upload box tells you which model is answering.

## Choosing a backend

Everything is driven by `.env`:

```ini
LLM_BACKEND=groq        # or: ollama
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
OLLAMA_MODEL=llama3.1:8b
```

|  | `ollama` | `groq` |
|---|---|---|
| Where the resume goes | stays on this machine | sent to Groq's API |
| Time for an 8-page CV | ~20-45 min | ~3 min (see below) |
| Resume text sent | 28,000 chars | 12,000 chars |
| Setup | `ollama pull llama3.1:8b` (4.9 GB) | an API key |

### Llama 3.1 is not available on Groq

Groq has retired it. The only Meta models left are `llama-prompt-guard-2`
classifiers with a 512-token context, which cannot do this job. The available
chat models are `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`
and `groq/compound`. `GROQ_MODEL` defaults to `openai/gpt-oss-120b`.

For actual Llama 3.1, use `LLM_BACKEND=ollama` — that path still runs
`llama3.1:8b` locally and reads the full 28k characters.

### The free tier is the bottleneck, not the model

A single Groq call answers in about a second. But the free tier meters **8,000
tokens per minute** on every gpt-oss / qwen model, and this pipeline makes six
calls. So a full report takes ~3 minutes, nearly all of it spent waiting out the
rate limit. `ask_llm()` handles `429` by reading the `retry-after` header and
sleeping, so it recovers rather than failing.

Two knobs if that matters:

- `GROQ_MAX_RESUME_CHARS` (default 12000) — smaller input, fewer tokens per call.
  At 28000 a single request exceeds the whole per-minute budget and returns 413.
- Upgrade to Groq's Dev Tier, which lifts the TPM cap and makes the whole report
  land in seconds.

## How it works

```
browser  ──POST /api/analyze (multipart)──►  FastAPI
                                              │
                                    pypdf / python-docx  →  plain text
                                              │
                                    ┌─────────▼──────────┐
                                    │  call 1: what role │  detect role + seniority
                                    │  is this resume    │  + the sections that role
                                    │  aiming at?        │  is expected to have
                                    └─────────┬──────────┘
                                              │
                    ┌───────────┬─────────────┼─────────────┬───────────┐
                    ▼           ▼             ▼             ▼           │
              Experience    Evidence       Growth     Completeness      │  4 sequential
                    │           │             │             │           │  JSON-mode calls
                    └───────────┴──────┬──────┴─────────────┘           │
                                       │                                │
                              call 6: recommended action  ◄─────────────┘
                                       │
browser  ◄────────── SSE: role, check ×4, done ──────────┘
```

Each result card appears in the browser the moment its check finishes, so you are
not staring at a spinner for the whole run.

## The four checks

| Check | What the model is told to look at |
|---|---|
| **Experience** | The project/job descriptions (real work vs. duty lists) and the dates (tenure, gaps). No dates anywhere caps the score at 55. |
| **Evidence** | Every claimed skill is pulled out, then looked for inside the experience section. Each is graded `strong` (used with context/outcome), `moderate` (mentioned only), `claimed` (skills-list only, never used). The score is how much of the list is actually proven. |
| **Growth** | Roles sorted by date, titles compared over time. Scope growth counts even without a promotion; a flat title for years scores low. One job only is judged on growth inside that job. |
| **Completeness** | Not a fixed checklist — call 1 decides which sections *this role* needs, and those are what get checked, plus universals (contact, summary, skills, dated experience, education). |

Overall score = mean of the four. `>= 70` green, `50-69` amber, `< 50` red.

### Why Evidence is not scored by the model

An 8B model asked to "list every skill" will confidently add technologies the
resume never mentioned — in testing it invented Java, C++, Node.js, Cassandra and
Spark, which inflated the denominator and dragged the score down for no reason.
So `ground_evidence()` in [app.py](app.py) drops any skill whose name does not
literally appear in the resume text, de-duplicates the rest, and then computes the
score in Python:

```python
SKILL_WEIGHT = {"strong": 1.0, "moderate": 0.6, "claimed": 0.15}
score = 100 * sum(weights) / len(skills)
```

The model still writes the sentence; the number always matches the chips on
screen. Adjust `SKILL_WEIGHT` if `claimed` skills feel over-punished.

## Speed

The six calls run **sequentially**, on purpose. They used to be issued
concurrently, but Ollama serialises requests anyway, so the last check just sat in
the queue until its read timeout fired and killed the whole report. Results stream
in as each check lands, so the table fills progressively either way.

- `ollama` - 20-45 min for a long CV. `OLLAMA_NUM_PARALLEL=2` before starting
  Ollama helps, as does a smaller quantised tag.
- `groq` - ~3 min on the free tier, almost entirely rate-limit waiting.

## Files

- [app.py](app.py) - extraction, prompts, backend dispatch, SSE endpoint
- [.env](.env) - backend choice, API key, model names (git-ignored)
- [static/index.html](static/index.html) - the whole frontend, no build step

## Frontend

Three views in one file, switched client-side: **upload -> analysing -> report**.
Tailwind, Lora/Inter and Phosphor icons load from CDNs, so the page needs an
internet connection even when `LLM_BACKEND=ollama`.

While the report streams in, each check shows live in a stepper, then its card
eases into the table with the score bar animating to value. Every row carries a
**Show full justification** drawer holding the model's `reasoning` paragraph plus
the evidence chips; Evidence chips can be filtered by Proven / Mentioned /
Unproven, and Completeness lists Present and Not-found sections. The headline
score counts up on completion. **Export Report** prints, with every drawer forced
open in the print stylesheet so nothing is hidden in the PDF.

Adding a fifth check means appending to `CHECKS` in [app.py](app.py) and adding
its key to `ORDER` plus an entry in `META` in the HTML.

## Tuning

Backend, model and key live in `.env` (see above). The rest is at the top of
[app.py](app.py):

```python
NUM_CTX = 16384          # ollama context window
MAX_RESUME_CHARS         # 28000 on ollama, 12000 on groq
SKILL_WEIGHT             # how harshly unproven skills are scored
```

The rubrics live in the `CHECKS` list — edit a `prompt` string to change how a
dimension is judged. To add a fifth check, append a dict to `CHECKS` and add its
key to `ORDER` in `static/index.html`.
