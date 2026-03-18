# TODO / Design Notes

## Project Philosophy & Intent

- **Goal**: Build a **cognitive operating system** — a persistent, digital extension of your memory and reasoning.
- **Paradigm**: Knowledge is a **living semantic structure**, not static files. The system organizes information into **semantic tunnels** (trajectories of thought) instead of folders.
- **Differentiator**: Combines **semantic understanding** (vector database), **temporal reasoning** (time-aware resurfacing and decay), and **multi-model intelligence** (routing queries to different LLMs).

---

## System Architecture (Cloud-Only, Zero-Cost)

- **Interface**: Telegram bot (primary and initial interface).
- **Compute / Orchestrator**: Python backend (FastAPI/Flask) running on **Koyeb Free Tier**.
- **Monitoring**: **UptimeRobot** hits a simple “alive” route every few minutes to keep the Koyeb app from sleeping.
- **Embeddings / Meaning**: **Google Gemini Embeddings** (`models/text-embedding-004`).
- **Vector Database (Long-Term Memory)**: **Pinecone** (Free Tier).
- **LLMs (Brains)**:
  - **Fast chat**: Groq `llama-3.1-8b-instant`.
  - **Deep thinking**: NVIDIA NIM `meta/llama-3.1-nemotron-70b-instruct`.
  - **Backup / context**: Google Gemini `gemini-1.5-flash` where helpful.
- **Local compute**: No local LLMs (no Ollama); the server only runs orchestration logic and calls cloud APIs.

---

## Technical Constraints & Optimizations

- **Koyeb Free Tier RAM (~512MB)**:
  - Cannot host local models; all LLM / embedding work must use external APIs.
  - Keep in-process state light; avoid loading big libraries unnecessarily.
- **Ephemeral storage on Koyeb**:
  - Treat local disk as temporary only.
  - Persist durable memory exclusively in **Pinecone**.
- **Rate limits / concurrency**:
  - Designed for **one primary user** (plus maybe 1–2 others).
  - Share free-tier limits across Groq, NVIDIA, Google → keep prompts small, context limited (e.g., **max 3 retrieved documents per query**).

---

## Codebase Structure (Planned)

```text
/exocortex
├── main.py              # Entry point. Runs Flask + Telegram Bot thread.
├── requirements.txt     # Dependencies.
├── .env.example         # Template for keys.
└── src/
    ├── config.py        # Loads environment variables & API keys.
    ├── memory.py        # Handles Pinecone & Gemini Embeddings.
    ├── brains.py        # Handles Groq, NVIDIA, & Gemini models + router logic.
    ├── telegram_bot.py  # Telegram handlers (messages, files).
    └── utils.py         # Helpers (text splitting, error handling).
```

This is the “engine” that powers cortexa.

---

## Implementation Roadmap (Engine)

### 1. Setup & Configuration

- **Initialize project**
  - Create the directory structure above.
  - Add `requirements.txt` with core deps: `python-telegram-bot`, `flask` or `fastapi`, `pinecone-client`, `langchain-*`, `python-dotenv`, `google-generativeai`, `groq`, `langchain-nvidia-ai-endpoints`, etc.
- **Config module (`src/config.py`)**
  - Load all environment variables safely.
  - Provide typed accessors for critical settings (API keys, index names, model IDs).
- **Environment variables**
  - `TELEGRAM_TOKEN`
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME` (default: `mysecondmind`)
  - `GOOGLE_API_KEY`
  - `GROQ_API_KEY`
  - `NVIDIA_API_KEY`

---

### 2. Memory Layer (`src/memory.py`)

- **Connect to Pinecone**
  - Initialize client with `PINECONE_API_KEY`.
  - Ensure index `mysecondmind` exists (dimension `768`, metric `cosine`).
- **Connect to embeddings**
  - Initialize `GoogleGenerativeAIEmbeddings` with `models/text-embedding-004`.
- **Create `MemoryManager`**
  - `add_memory(text, metadata)`: embed with Gemini, upsert into Pinecone.
  - `recall_context(query, k=3)`: embed query, search Pinecone, return top-k memories.

---

### 3. Brains & Router (`src/brains.py`)

- **Initialize models**
  - `fast_brain = ChatGroq(model="llama-3.1-8b-instant")`
  - `smart_brain = ChatNVIDIA(model="meta/llama-3.1-nemotron-70b-instruct")`
  - Optional: Gemini chat client for backup/contextual tasks.
- **Router function**
  - `route_query(query, context, mode_hint=None)`:
    - If query contains planning / deep-analysis cues (`plan`, `roadmap`, `analyze`, `debug`, etc.) or `mode_hint == "deep"`, use `smart_brain`.
    - Otherwise, use `fast_brain`.
    - Always inject retrieved `context` from `MemoryManager` into the prompt.
- **Prompt structure**
  - Use a consistent template:
    - System: “You are Exocortex, a cognitive OS.”
    - Context: `{retrieved_memory}` (up to 3 docs).
    - User: `{user_query}`.

---

### 4. Telegram Interface (`src/telegram_bot.py`)

- **Handlers**
  - `/start`: Welcome message explaining cortexa and how to talk to it.
  - Text handler:
    - Receive text.
    - Call `MemoryManager.recall_context`.
    - Call `route_query` with query + context.
    - Save new result/interaction back via `add_memory`.
    - Reply to user.
  - Document handler (PDFs):
    - Download PDF to temp storage.
    - Extract text.
    - Chunk with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200).
    - Call `add_memory` on each chunk with appropriate metadata.

---

### 5. Server & Orchestration (`main.py`)

- **Flask (or FastAPI) server**
  - Expose `/` health route returning `"Alive"` for UptimeRobot and Koyeb.
- **Threading model**
  - Run Flask app in a background thread.
  - Run Telegram bot `run_polling()` in the main thread.
- **Deployment**
  - Configure Koyeb with:
    - Start command (e.g., `python main.py`).
    - Environment variables from the “Shopping List”.

---

### 6. Cursor AI Coding Guidelines

- **Async where helpful**
  - Prefer async handlers in Telegram where library support is robust, especially for network-bound work (Pinecone / LLM calls).
- **Error handling**
  - Wrap external API calls (Groq, NVIDIA, Google, Pinecone) in `try/except`:
    - Gracefully handle rate limits, timeouts, and transient failures.
    - Return a friendly message to the user and log details internally.
- **Context control**
  - Never pull more than **3 documents** from Pinecone for a single query to keep latency and token usage low.
- **Chunking strategy**
  - Use `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200` for PDF/text ingestion.

---

## Security and Safety

- Safeguard from phishing links or exposing anything.
- Prompt injection security.
- Script injections.

## Orchestration Layer

When you send:

- Link
- Image
- Text
- Task
- Deadline

Our orchestration layer decides: “Which tool should I call?”

Example:

- If link → call metadata extractor.
- If image → call vision model.
- If task → call reminder scheduler.
- If research → generate summary + tags.

## Chat Interfaces (Telegram and WhatsApp)

- **Telegram (MVP, ingestion + retrieval)**
  - Create a Telegram bot via BotFather and connect it to the backend.
  - Use Telegram as the **primary channel** for both:
    - Ingesting memories (notes, links, ideas, tasks) in simple English.
    - Querying existing memories in simple English (“Ask it about anything and it should understand and fetch what you want”).
  - Implement intent detection to decide:
    - When a message should **store** a new memory.
    - When it should **retrieve** and summarize relevant memories.
  - Ensure responses are conversational and context-aware inside the chat thread.
- **WhatsApp (Phase 2, ingestion + retrieval)**
  - Integrate WhatsApp via Baileys or a similar library.
  - Mirror the same behaviors as Telegram:
    - Natural-language ingestion and retrieval.
    - Shared orchestration + semantic retrieval pipeline.
  - Handle basic authentication / access control so only the owner can talk to their second mind via WhatsApp.

### Voice Notes and Passive Capture

- Support **voice notes** as a first-class input:
  - Telegram/WhatsApp voice message → local transcription (e.g., Whisper) → ingestion pipeline.
  - Extract intent, entities, and tasks from transcribed text.
- Encourage a “dump now, structure later” workflow:
  - Chat/voice is optimized for fast capture (“Dump Mode”).
  - Web dashboard is optimized for review, visualization, and reflection (“Review Mode”).

## Memory Intelligence Layer

Instead of hard-coded SQL resurfacing, our system should:

- Decide which memories are important.
- Prioritize resurfacing based on:
  - Tags
  - Importance
  - Recency
  - Your behavior

This is much smarter than cron-only logic.

### Forgetting and Pruning

- Design a **decay model** for memory activation:
  - Lower scores for items that are rarely accessed or consistently marked “not relevant”.
  - Decide thresholds for:
    - Keeping items in active recall.
    - Archiving or down-weighting them.
- Integrate pruning into:
  - Tunnel maintenance (remove or de-emphasize dead-end items).
  - Resurfacing and RAG context selection.

### Emotional Valence and Friction

- Add optional metadata fields:
  - `valence` (positive/negative/neutral sentiment).
  - `friction` (how “stuck” or effortful a memory/problem feels).
- Use LLM or heuristics to estimate these when ingesting memories.
- Feed valence and friction into:
  - Resurfacing priority (e.g., highlight high-friction, high-importance problems).
  - Analytics about what types of work consume the most cognitive energy.

### Notifications and Resurfacing Delivery

- Design a **notification strategy** for:
  - Daily digest (summary of new memories, topics, and tunnels).
  - Resurfacing prompts (high-score memories pushed proactively).
  - General nudges or insights from the reflection engine.
- Implement a notification pipeline that can:
  - Format resurfaced items and summaries into concise chat messages.
  - Send them via:
    - Telegram (MVP).
    - WhatsApp (once Phase 2 integration is done).

## Deadline Detection Agent

Instead of you writing regex for dates, our agent/system can:

- Extract date.
- Confirm timezone.
- Decide reminder frequency.
- Schedule follow-ups.

Example:

You send: “Hackathon submission March 8”.

Agent:

- Detects date.
- Adds 2-day-before reminder.
- Adds same-day reminder.
- Confirms with you.

### Reminder Notifications

- Connect detected deadlines to a **reminder scheduler**.
- Ensure reminders are delivered as:
  - Telegram messages in the relevant chat/thread.
  - WhatsApp messages once Phase 2 integration is ready.
- Support multiple reminder patterns (e.g., “2 days before”, “same day morning”, “evening check-in”).

### Serenity / Nagging Threshold

- Add a configurable **serenity setting** for notification frequency and intrusiveness.
- Track how often the user:
  - Opens or interacts with digests and resurfacing prompts.
  - Ignores or dismisses them.
- Adapt notification cadence to stay helpful without becoming annoying.

## Product and UX Improvements

1. **Sharpen the MVP around one or two workflows**  
   Clarify the first “hero use cases”, for example:  
   - “Capture research ideas + see them again when they become relevant.”  
   - “Track all tasks/ideas with deadlines and get smart reminders + nightly summary.”  
   That will guide which parts of ingestion, resurfacing, and analytics must be rock-solid in v1 vs “nice to have later”.
2. **Better feedback loop and learning**  
   User feedback signals on resurfaced items:  
   - “Relevant / not relevant”, “remind later”, “snooze”, “archive this”.  
   Use these signals to update `priority_score` or a separate “usefulness score” so the system learns your taste over time instead of staying static.
3. **Richer temporal and UI views**  
   Add explicit time-based views:  
   - Timeline of memories with filters by topic/priority.  
   - “This week’s resurfaced items” vs “things you ignored.”  
   A home dashboard concept with:  
   - Today’s resurfaced items  
   - Deadlines approaching  
   - Short daily reflection snippet
4. **Stronger task/deadline integration**  
   Right now tasks + deadlines are mentioned, but you could:  
   - Make a first-class “Task Memory” type with fields like `status`, `estimated_effort`, `project`.  
   - Integrate (optionally) with a calendar or time-blocking view.  
   - Use resurfacing not just for “remember this idea” but “nudge you to finish tasks before they decay.”
5. **Capture from multiple channels**  
   To reduce friction / fragmentation further:  
   - Browser extension or bookmarklet to send current tab as a memory.  
   - Email-to-memory address (forward emails into the system).  
   - Simple CLI or chat-like interface to dump thoughts quickly.  
   Each reduces the chance that information lives outside your second mind.
6. **Early, pragmatic knowledge graph**  
   The README lists knowledge graph as a “future extension”. You could:  
   - Start lightweight entity extraction (people, projects, tools, concepts) and store them as structured tags/relations.  
   - Provide simple views like: “All memories related to <Person X>” or “All memories related to <Project Y>”.  
   This would make cross-domain synthesis more visible earlier.
7. **Personal metrics and goals**  
   Tie analytics to user-set goals:  
   - For example, “This month I want 40% of my cognitive time on GATE prep.”  
   Compare actual topic distribution vs goal.  
   This directly attacks attention drift by giving you a “compass”, not just a retrospective.
8. **Privacy, security, and portability**  
   Since this is personal cognition, clarify:  
   - Encryption at rest for the database (or at least support it).  
   - Export/import of all data (JSON or similar) so you’re not locked in.  
   - Clear story for “if I change machines, how do I take my mind with me?”

## Persistent Memory and Tunnels

- Implement persistent memory.

### Tunnels: Dynamic Semantic Pathways

1. **What a tunnel is**  
   - A tunnel is a dynamic, evolving semantic pathway through your memory.  
   - It is not a folder, a tag, a project, or a static category.  
   - It is a continuously updating stream of related thoughts across time.

2. **Why tunnels are needed**  
   - Folders are rigid.  
   - Tags are shallow.  
   - Projects are temporary.  
   - Your thinking is not linear.  
   - Example: you think about FPGA, ASCON, AI inference, hardware security, and memristor PUF. These are not separate folders; they form a conceptual pathway. That pathway is a tunnel.

3. **How a tunnel is formed**  
   Technically, a tunnel emerges when:  
   - Multiple memory items cluster in embedding space.  
   - They share semantic similarity.  
   - They appear repeatedly over time.  
   - They intersect across sessions.  
   Instead of you manually creating something like “Hardware Security”, the system detects a recurring semantic cluster. That cluster becomes a tunnel.

4. **What a tunnel contains**  
   A tunnel includes:  
   - Core theme vector (centroid of embeddings).  
   - Related memory items.  
   - Timeline of evolution.  
   - Idea branches.  
   - Deadlines connected to it.  
   - Activity frequency.  
   Think of it like a living research corridor.

5. **Example tunnel (personalized)**  
   You save:  
   - FPGA secure inference idea.  
   - ASCON lightweight crypto note.  
   - Medical edge inference architecture.  
   - Hackathon draft.  
   - Memristor-PUF security idea.  
   The system sees:  
   - Embedding similarity increasing.  
   - Temporal clustering increasing.  
   - Recurrent keywords increasing.  
   A tunnel is created, for example: “Hardware-Accelerated AI Security Tunnel”.  
   Now any time you add related memory, search similar topics, or discuss related ideas, it gets absorbed into this tunnel.

6. **What makes tunnels powerful**  
   Tunnels allow:  
   - Long-term idea continuity.  
   - Cross-domain linking.  
   - Automatic project grouping.  
   - Intellectual evolution tracking.  
   Instead of remembering individual notes, you remember the trajectory of your thinking.

7. **Tunnel intelligence**  
   Each tunnel tracks:  
   - Growth rate.  
   - Dormancy.  
   - Deadline pressure.  
   - Convergence with other tunnels.  
   - Drift into new themes.  
   Example: the system detects an FPGA tunnel converging with a sustainability tunnel. That is idea synthesis and innovation.

8. **Tunnel visualization (dashboard idea)**  
   Each tunnel appears as:  
   - A timeline graph.  
   - A branching map.  
   - A density heatmap.  
   - A priority meter.  
   You do not see isolated notes; you see idea ecosystems.

9. **How tunnels are implemented (technical core)**  
   - Step 1: Store embeddings for all memory items.  
   - Step 2: Periodically cluster embeddings (k-means / hierarchical clustering).  
   - Step 3: Identify stable clusters over time.  
   - Step 4: Assign the cluster centroid as the tunnel vector.  
   - Step 5: Track cluster evolution.  
   Tunnels are dynamic vector clusters across time.

10. **Deep insight**  
    - Folders organize data.  
    - Tunnels organize thought trajectories.  
    - That is the difference.

---

## Diary and Longitudinal Profile

- **Diary entries**
  - Implement periodic jobs (weekly, monthly, yearly) that:
    - Fetch memories, tunnels, and key events for the period.
    - Compute high-level stats: dominant topics, active tunnels, milestones.
    - Call a deep-thinking LLM (NVIDIA) to generate a **narrative diary entry**:
      - “What you focused on.”
      - “What changed compared to last period.”
      - “Emerging or fading themes.”
  - Store each summary as a `diary_entry` memory with metadata (`period_start`, `period_end`, `granularity = daily/weekly/monthly/yearly`).
  - Deliver monthly and yearly diaries as Telegram messages for reflection.
- **Personal profile**
  - Maintain a derived, high-level profile object:
    - Core interests and tunnels.
    - Typical focus split (research vs building vs planning).
    - Long-term trends in attention drift and convergence.
  - Use this profile as:
    - Extra context for diary prompts.
    - A compact, queryable “profile of you” inside cortexa.