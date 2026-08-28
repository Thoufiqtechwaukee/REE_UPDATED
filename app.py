"""
Resume Depth - resume analyzer.

Frontend (static/index.html) uploads a resume -> this server extracts the text ->
an LLM scores 4 dimensions -> results stream back to the browser one card at a
time.

The model is pluggable via .env:
    LLM_BACKEND=ollama   local, private, slow  (llama3.1:8b)
    LLM_BACKEND=groq     hosted, fast, metered (see GROQ_MODEL)

Run:  uvicorn app:app --reload --port 8000
"""

import asyncio
import io
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# "groq" = hosted and fast; "ollama" = local and private. See .env.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MODEL = GROQ_MODEL if LLM_BACKEND == "groq" else OLLAMA_MODEL

# Ollama can now point at a remote GPU, so the backend name no longer tells us
# whether the resume stays on this machine. Only the host does.
IS_LOCAL = LLM_BACKEND != "groq" and bool(
    re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:|/|$)", OLLAMA_URL))
HOST = re.sub(r"^https?://", "", GROQ_URL if LLM_BACKEND == "groq" else OLLAMA_URL).split("/")[0]
NUM_CTX = 16384              # ollama only: must hold resume + prompt + reply

# Local Ollama only cares about context size, so it can read a whole 8-page CV.
# Groq's free tier meters tokens per minute (8000 on the gpt-oss models), and a
# 28k-char resume alone is ~7k tokens - one request would blow the whole minute.
MAX_RESUME_CHARS = int(os.getenv(
    "GROQ_MAX_RESUME_CHARS" if LLM_BACKEND == "groq" else "OLLAMA_MAX_RESUME_CHARS",
    "12000" if LLM_BACKEND == "groq" else "28000"))

app = FastAPI(title="Resume Depth")


# --------------------------------------------------------------------------
# 1. Text extraction
# --------------------------------------------------------------------------

def extract_text(filename: str, blob: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(blob))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".docx":
        import docx
        doc = docx.Document(io.BytesIO(blob))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        text = "\n".join(parts)
    elif ext in (".txt", ".md"):
        text = blob.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use PDF, DOCX, TXT or MD.")

    # squeeze blank lines / trailing spaces so the model sees clean structure
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError("No text found in this file (is it a scanned image PDF?).")

    # A plain head-truncation on a long CV throws away Education and the earliest
    # jobs, which is exactly what Completeness and Growth need. Keep both ends.
    if len(text) > MAX_RESUME_CHARS:
        head = int(MAX_RESUME_CHARS * 0.65)
        tail = MAX_RESUME_CHARS - head
        text = text[:head] + "\n\n[... middle of resume omitted ...]\n\n" + text[-tail:]
    return text


# --------------------------------------------------------------------------
# 2. Ollama plumbing
# --------------------------------------------------------------------------

# Generation can genuinely take many minutes for an 8B model on CPU, and Ollama
# queues requests, so a short read timeout just turns a slow answer into a crash.
LLM_TIMEOUT = httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=1800.0)


async def call_ollama(client: httpx.AsyncClient, system: str, user: str,
                      num_predict: int) -> str:
    resp = await client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": NUM_CTX,
                "num_predict": num_predict,
            },
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


async def call_groq(client: httpx.AsyncClient, system: str, user: str,
                    num_predict: int) -> str:
    resp = await client.post(
        f"{GROQ_URL}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            # reasoning models spend part of this budget before the JSON starts
            "max_tokens": max(num_predict * 2, 4096),
        },
        timeout=LLM_TIMEOUT,
    )
    if resp.status_code == 429:                          # free tier is metered per minute
        raise RateLimited(float(resp.headers.get("retry-after", 20)))
    if resp.status_code >= 400:                          # surface Groq's own message
        try:
            detail = resp.json()["error"]["message"]
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(f"Groq {resp.status_code}: {detail}")
    return resp.json()["choices"][0]["message"]["content"]


class RateLimited(Exception):
    def __init__(self, wait: float):
        self.wait = min(wait, 65.0)
        super().__init__(f"rate limited, waiting {self.wait:.0f}s")


async def ask_llm(client: httpx.AsyncClient, system: str, user: str,
                  num_predict: int = 1600) -> dict:
    """One JSON-mode call to whichever backend is configured. Returns a dict."""
    call = call_groq if LLM_BACKEND == "groq" else call_ollama
    last_error = None
    for attempt in range(4):
        try:
            return parse_json(await call(client, system, user, num_predict))
        except RateLimited as limit:
            last_error = limit                           # the minute window resets
            await asyncio.sleep(limit.wait)
        except Exception as exc:
            last_error = exc                             # one retry; models drift
            if attempt >= 1:
                break
    raise describe(last_error)


def parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)      # model wrapped it in prose
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return JSON: {content[:200]}")


def describe(exc: Exception) -> Exception:
    """httpx timeouts stringify to '', which reaches the UI as a blank error."""
    if str(exc):
        return exc
    return RuntimeError(f"{type(exc).__name__} (the model took too long to answer)")


def clamp_score(value) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 50
    return max(0, min(100, n))


def status_for(score: int) -> str:
    if score >= 70:
        return "pass"
    if score >= 50:
        return "warn"
    return "fail"


# 8B models happily invent skills the resume never claimed, which poisons both the
# chips and the ratio they are scored on. Keep only skills whose name literally
# appears in the resume, then compute the score ourselves from what survived, so
# the number always matches the chips on screen.
SKILL_WEIGHT = {"strong": 1.0, "moderate": 0.6, "claimed": 0.15}


def ground_evidence(data: dict, resume: str) -> dict:
    skills = data.get("skills")
    if not isinstance(skills, list):
        return data

    haystack = resume.lower()
    kept, seen = [], set()
    for item in skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        key = name.lower()
        # CVs that write their skills as prose bullets tempt the model into
        # returning a whole sentence as one "skill" - a real one is a short name.
        if not name or len(name) > 40 or key in seen or key not in haystack:
            continue
        seen.add(key)
        strength = item.get("strength")
        kept.append({
            "name": name,
            "strength": strength if strength in SKILL_WEIGHT else "moderate",
            "note": str(item.get("note") or ""),
        })

    if not kept:
        return data

    proven = sum(1 for s in kept if s["strength"] != "claimed")
    data["skills"] = kept
    data["score"] = round(100 * sum(SKILL_WEIGHT[s["strength"]] for s in kept) / len(kept))
    data["detail"] = f"{proven} of {len(kept)} claimed skills appear in the described work"
    return data


def ground_completeness(data: dict, resume: str) -> dict:
    """The model also calls things missing that are plainly in the document.

    A 'missing' item names the thing it is looking for, so take its proper nouns
    (Outsystems, APEX, LinkedIn, CBAP) and check them against the text. If every
    one is already there, the section is not missing and the claim is dropped.
    """
    missing = data.get("missing")
    if not isinstance(missing, list):
        return data

    text = resume.lower()
    kept = []
    for item in missing:
        label = str(item).strip()
        if not label:
            continue
        names = re.findall(r"\b[A-Z][A-Za-z0-9+#.\-]{2,}\b", label)
        if names and all(n.lower() in text for n in names):
            continue                                     # provably present
        kept.append(label)

    data["missing"] = kept
    return data


# --------------------------------------------------------------------------
# 3. The four checks
# --------------------------------------------------------------------------

SYSTEM = (
    "You are a strict technical recruiter reviewing a resume. "
    "You judge only what is written in the resume - never invent facts. "
    "Never write 'I' - address the candidate as 'you'. "
    "You reply with JSON only, no prose, no markdown fences."
)

ROLE_PROMPT = """Read the resume and identify what job the candidate is positioned for.

RESUME:
{resume}

Reply with JSON exactly like:
{{"role": "<job title, e.g. Backend Engineer>",
  "seniority": "<Intern|Junior|Mid-level|Senior|Lead|Manager>",
  "expected_sections": ["<section a resume for THIS role must have>", "..."]}}

Base "expected_sections" on what this specific role and seniority is judged on
(e.g. a data scientist needs projects with datasets and metrics; a manager needs
team size and scope; a fresher needs education and academic projects)."""


CHECKS = [
    {
        "key": "experience",
        "title": "Experience",
        "why": "The system reads the project content to see what work they have really "
               "done, and the dates to see how long they did it for.",
        "prompt": """Judge the DEPTH OF EXPERIENCE in this resume.

Detected role: {role} ({seniority})

RESUME:
{resume}

Rules:
- Read the actual project/job descriptions. Real engineering work (systems built,
  problems solved, decisions made) scores high. Vague duty lists score low.
- Read the dates. Long tenure and continuous history score high. Gaps, very short
  stints, or missing dates score low.
- No dates anywhere = cap the score at 55.

Reply with JSON:
{{"score": <0-100>,
  "verdict": "<one sentence addressed to the candidate as 'Your experience ...'>",
  "detail": "<one short sentence naming the concrete evidence you used>",
  "reasoning": "<3-4 sentences. Name the actual employers, projects and dates you
    read. Say what the work shows about their depth, and what specifically held the
    score back. Address the candidate as 'you'. Quote real phrases from the resume.>",
  "total_years": "<e.g. '4.5 years' or 'unclear'>"}}""",
    },
    {
        "key": "evidence",
        "title": "Evidence",
        "why": "We check that the skills they claim actually show up in the work they described.",
        "prompt": """Check whether the CLAIMED SKILLS are backed by the described work.

Detected role: {role} ({seniority})

RESUME:
{resume}

Steps:
1. List every skill/tool/technology the resume claims (skills section, summary, anywhere).
   Copy the names EXACTLY as written in the resume. Never add a technology that
   does not appear in the text above - a wrong name invalidates the whole check.
2. For each one, look for it inside the experience or project descriptions.
   - "strong"   = used in a described project WITH context or an outcome
   - "moderate" = mentioned in a project but with no detail
   - "claimed"  = listed only in the skills section, never used anywhere

Score = how much of the skill list is actually proven. Mostly "claimed" scores low.

Reply with JSON:
{{"score": <0-100>,
  "verdict": "<one sentence addressed to the candidate>",
  "detail": "<one short sentence, e.g. '9 of 14 claimed skills appear in project work'>",
  "reasoning": "<3-4 sentences. Name which skills are genuinely proven and where
    they were proven. Then name the ones that appear only in the skills list, and
    say what evidence would fix them. Address the candidate as 'you'.>",
  "skills": [{{"name": "<skill>", "strength": "strong|moderate|claimed",
               "note": "<where it was proven, or 'not used in any project'>"}}]}}

Include every skill you found, up to 20.""",
    },
    {
        "key": "growth",
        "title": "Growth",
        "why": "Do the job titles rise over time? Junior to Senior to Lead shows someone "
               "trusted with more. A flat line for eight years does not.",
        "prompt": """Judge CAREER GROWTH across the jobs in this resume.

Detected role: {role} ({seniority})

RESUME:
{resume}

Rules:
- Put the roles in date order and look at the titles. Rising titles
  (Intern -> Junior -> Senior -> Lead) score high.
- Growth also counts without a promotion: bigger scope, harder systems, leading
  people, owning more of the product over time.
- The same title with the same responsibilities for many years scores low.
- Only one job so far: judge growth INSIDE that job, and do not punish a short career.

Reply with JSON:
{{"score": <0-100>,
  "verdict": "<one sentence addressed to the candidate>",
  "detail": "<one short sentence naming the progression you saw, e.g. 'Junior Dev (2019) -> Senior Dev (2023)'>",
  "reasoning": "<3-4 sentences. Walk the roles in date order with their titles and
    years. Say where scope or responsibility widened, and where it plateaued. If
    the career is short, say so plainly instead of punishing it. Address the
    candidate as 'you'.>",
  "trajectory": "<rising|steady|flat|unclear>"}}""",
    },
    {
        "key": "completeness",
        "title": "Completeness",
        "why": "Are the basic parts there - a summary, a skill list, education, certificates. "
               "Missing pieces cost the candidate nothing to fix.",
        "prompt": """Check whether this resume has all the parts expected FOR ITS ROLE.

Detected role: {role} ({seniority})
Sections this role is expected to have: {expected_sections}

RESUME:
{resume}

Also always check the universals: contact details, professional summary, skills
list, work experience with dates, education. Certifications and links (GitHub /
portfolio / LinkedIn) count when the role expects them.

Score = share of expected parts that are present and usable.

Reply with JSON:
{{"score": <0-100>,
  "verdict": "<one sentence addressed to the candidate>",
  "detail": "<one short sentence>",
  "reasoning": "<3-4 sentences. Say which expected sections you found and where.
    Then name what is absent and why it matters for THIS role specifically.
    Address the candidate as 'you'.>",
  "present": ["<section found>", "..."],
  "missing": ["<expected section that is absent>", "..."]}}""",
    },
]


ACTION_PROMPT = """You reviewed a resume for a {role} ({seniority}) and scored it:

{summary}

Write the single highest-value fix for this candidate - start with the weakest
area. Two or three sentences, plain language, addressed to the candidate as "you".
Say what to do, not what is wrong.

Reply with JSON: {{"action": "<the advice>"}}"""


# --------------------------------------------------------------------------
# 4. Streaming endpoint
# --------------------------------------------------------------------------

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def preflight(client: httpx.AsyncClient) -> str:
    """Fail with a useful sentence before promising the user a report."""
    if LLM_BACKEND == "groq":
        if not GROQ_API_KEY:
            return "GROQ_API_KEY is not set. Add it to the .env file next to app.py."
        try:
            resp = await client.get(f"{GROQ_URL}/models",
                                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                                    timeout=15.0)
        except Exception as exc:
            return f"Could not reach Groq: {describe(exc)}"
        if resp.status_code == 401:
            return "Groq rejected the API key in .env (401). Check GROQ_API_KEY."
        if resp.status_code >= 400:
            return f"Groq returned {resp.status_code}: {resp.text[:160]}"
        names = [m["id"] for m in resp.json().get("data", [])]
        if GROQ_MODEL not in names:
            return (f"Groq has no model '{GROQ_MODEL}'. Available: "
                    + ", ".join(sorted(names)[:8]))
        return ""

    try:
        tags = (await client.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)).json()
    except Exception:
        return f"Ollama is not reachable at {OLLAMA_URL}. Start it and try again."
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL.split(":")[0]) for n in names):
        return f"Model '{OLLAMA_MODEL}' is not installed. Run:  ollama pull {OLLAMA_MODEL}"
    return ""


async def run_analysis(resume: str):
    async with httpx.AsyncClient() as client:
        problem = await preflight(client)
        if problem:
            yield sse("error", {"message": problem})
            return

        yield sse("step", {"message": "Reading the resume..."})
        try:
            role_info = await ask_llm(client, SYSTEM, ROLE_PROMPT.format(resume=resume))
        except Exception as exc:
            yield sse("error", {"message": f"Model call failed: {exc}"})
            return

        role = role_info.get("role") or "Professional"
        seniority = role_info.get("seniority") or "Mid-level"
        expected = role_info.get("expected_sections") or []
        yield sse("role", {"role": role, "seniority": seniority})

        ctx = {
            "resume": resume,
            "role": role,
            "seniority": seniority,
            "expected_sections": ", ".join(expected) if expected else "(use your own judgement)",
        }

        yield sse("step", {"message": f"Scoring against a {role} profile..."})

        async def run_check(check):
            # the skills array on a long CV is the one reply that can run long
            budget = 3000 if check["key"] == "evidence" else 1800
            data = await ask_llm(client, SYSTEM, check["prompt"].format(**ctx), budget)
            if check["key"] == "evidence":
                data = ground_evidence(data, resume)
            elif check["key"] == "completeness":
                data = ground_completeness(data, resume)
            score = clamp_score(data.get("score"))
            return {
                "key": check["key"],
                "title": check["title"],
                "why": check["why"],
                "score": score,
                "status": status_for(score),
                "verdict": data.get("verdict") or "",
                "detail": data.get("detail") or "",
                "reasoning": data.get("reasoning") or "",
                "extra": {k: v for k, v in data.items()
                          if k in ("skills", "present", "missing", "total_years", "trajectory")},
            }

        # Sequential on purpose. Ollama serialises requests anyway, so firing all
        # four at once buys no speed - it just leaves the last one sitting in the
        # queue until its read timeout fires.
        scored = []
        for check in CHECKS:
            try:
                result = await run_check(check)
                scored.append(result)
            except Exception as exc:
                result = {
                    "key": check["key"],
                    "title": check["title"],
                    "why": check["why"],
                    "score": 0,
                    "status": "error",
                    "verdict": "This check could not be completed.",
                    "detail": str(describe(exc))[:180],
                    "reasoning": "",
                    "extra": {},
                }
            yield sse("check", result)

        if not scored:
            yield sse("error", {"message": "Every check failed - is Ollama still running?"})
            return

        overall = round(sum(r["score"] for r in scored) / len(scored))

        summary = "\n".join(f"- {r['title']}: {r['score']}/100 - {r['verdict']}" for r in scored)
        try:
            action = (await ask_llm(client, SYSTEM, ACTION_PROMPT.format(
                role=role, seniority=seniority, summary=summary), 400)).get("action", "")
        except Exception:
            action = ""

        yield sse("done", {"score": overall, "status": status_for(overall), "action": action})


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    blob = await file.read()
    try:
        resume = extract_text(file.filename or "resume.pdf", blob)
    except Exception as exc:
        message = str(exc)

        async def fail():
            yield sse("error", {"message": message})

        return StreamingResponse(fail(), media_type="text/event-stream")

    return StreamingResponse(
        run_analysis(resume),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/backend")
async def backend():
    return {"backend": LLM_BACKEND, "model": MODEL, "local": IS_LOCAL, "host": HOST}


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
