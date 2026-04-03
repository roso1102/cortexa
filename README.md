# 🧠 cortexa

### Your self-hosted cognitive memory engine

---

## 1. Overview

cortexa is a self-hosted cognitive system designed to function as an externalized structured memory layer for an individual.

It ingests unstructured inputs (text, links, images), transforms them into structured knowledge objects, encodes them into semantic space, tracks them temporally, and resurfaces them intelligently.

The system is designed around one principle:

> Memory without structure is storage.
> Memory with structure, time, and intelligence becomes cognition.

---

## 2. Problem Statement

Modern knowledge workers face three systemic failures:

1. **Cognitive fragmentation** — ideas are scattered across chats, notes, bookmarks.
2. **Context decay** — saved content loses relevance without resurfacing.
3. **Attention drift** — focus shifts without awareness of pattern.

Traditional tools (Notion, Obsidian, Google Docs) provide storage and manual retrieval.

They do not:

* Model semantic similarity.
* Understand priority or deadlines.
* Resurface information intelligently.
* Analyze cognitive trends over time.

cortexa addresses these limitations.

---

## 3. System Philosophy

cortexa mirrors biological cognition:

| Biological System | Digital Equivalent              |
| ----------------- | ------------------------------- |
| Sensory Input     | Ingestion Layer                 |
| Hippocampus       | Structured Storage + Embeddings |
| Prefrontal Cortex | LLM Reasoning Layer             |
| Dopamine System   | Priority Scoring                |
| Circadian Recall  | Scheduled Resurfacing           |
| Meta-cognition    | Trend & Reflection Engine       |

This is not a note-taking system.

This is a cognitive augmentation system.

---

## 4. Core Capabilities

### 4.1 Intelligent Ingestion

Supports:

* Plain text
* URLs
* Images
* Tasks and deadlines
* Research notes
* Ideas
* Chat messages from Telegram / WhatsApp

The system automatically:

* Classifies content type
* Extracts metadata
* Generates summary
* Detects deadlines
* Assigns priority
* Tags semantically

---

### 4.2 Structured Memory Encoding

Each memory object contains:

* `raw_content`
* `content_type`
* `title`
* `summary`
* `tags`
* `deadline`
* `priority_score`
* `embedding_vector`
* `created_at`
* `last_accessed`
* `resurfaced_count`

Embeddings enable semantic recall rather than keyword matching.

Internally, the system can support different **memory regimes**:

* **Short-term working memory** — very recent interactions and context windows (primarily in the LLM layer).
* **Long-term semantic memory** — all persisted items in the vector database (pgvector).
* **Structured memory** — higher-level objects like tunnels, tasks, and entities built on top of raw items.

---

### 4.3 Semantic Retrieval

Query processing pipeline:

1. Convert query to embedding.
2. Perform vector similarity search in the vector database (pgvector).
3. Optionally perform RAG-style retrieval: assemble the most relevant memories as context.
4. Rerank and summarize using the LLM reasoning layer.
5. Return structured, relevant results.

Supports natural language recall:

> “That FPGA ASCON medical inference idea.”

In practice, you should be able to ask in simple English, through chat (Telegram/WhatsApp or web), and the system will:

* Infer whether you are trying to **store** a new memory or **retrieve** existing ones.
* Interpret your intent from natural language.
* Fetch and present the most relevant memories, not just keyword matches.

---

### 4.4 Temporal Intelligence

Memory is not static.

Each memory item evolves based on:

* Age
* Access frequency
* Deadline proximity
* Priority score
* Thematic alignment with recent activity

Resurfacing logic combines:

```
resurface_score =
  priority_weight
+ aging_factor
+ semantic_relevance_to_recent_activity
- resurfaced_penalty
```

This creates natural spaced repetition without explicit scheduling.

---

### 4.5 Reflective Analytics

Nightly system analysis provides:

* Topic distribution
* Focus concentration
* Deadline pressure
* Thematic drift
* Cognitive trend evolution

Example output:

> Today’s focus:
> 60% Hardware Security
> 25% GATE Preparation
> 15% Miscellaneous
>
> Trend: Increasing convergence between FPGA and AI security topics.

This enables meta-cognition.

---

### 4.6 Notifications and Delivery

cortexa is not only pull-based search; it can also **push** information back to you.

Notification channels:

* **Telegram** (primary in early phases).
* **WhatsApp** (added later).

Types of notifications:

* **Daily digest** — short summary of today’s created memories, topic distribution, and key tunnels or themes.
* **Resurfacing prompts** — selected high-score memories delivered as contextual recall messages.
* **Reminders** — deadline-based pings for tasks and time-sensitive memories.

Notifications are delivered as chat messages through your personal bots, so your second mind can proactively talk to you.

---

## 5. Architecture

### 5.1 High-Level Architecture (The Engine)

```
You (Telegram chat)
        ↓
Telegram Bot (interface)
        ↓
cortexa Backend on Koyeb (FastAPI)
        ↓
LLM Layer
  - Groq: Llama-3.1-8B (fast chat)
  - NVIDIA: Nemotron-70B (deep thinking)
        ↓
Memory Layer
  - Google Gemini Embeddings
  - Pinecone (vector database)
```

- **Interface (How you talk to it)**: Telegram Bot.
- **Hosting (The body)**: Koyeb (Free Tier) – chosen over Render for better free-tier performance and stability.
- **Brain (LLMs)**:
  - Groq (Llama-3.1-8B) for fast daily chat and lightweight reasoning.
  - NVIDIA (Nemotron-70B) for complex reasoning, planning, and heavier analysis.
- **Memory (Embeddings)**: Google Gemini Embeddings (Free API).
- **Storage (Database)**: Pinecone (Free Tier) as the vector store for all memory embeddings.

Other capture channels (WhatsApp, browser extension, email, CLI, etc.) can still be added later, but Telegram is the primary interface for the first version.

---

### 5.2 Technology Stack

**Interface**

* Telegram Bot (primary early interface for ingestion + retrieval).

**Backend / Hosting**

* FastAPI backend.
* Deployed on **Koyeb** (Free Tier).

**Inference Engine (LLMs)**

* **Groq** – Llama-3.1-8B for fast, low-latency chat.
* **NVIDIA Build** – Nemotron-70B for deep thinking and complex planning.

**Embeddings / Memory Encoding**

* **Google Gemini Embeddings API** for embedding all memories and queries.

**Vector Database / Storage**

* **Pinecone** (Free Tier) as the main vector database.

---

## 5.3 Setup “Shopping List”

Before running cortexa, you need API keys from these 5 services:

1. **Telegram**
   - Talk to `@BotFather`, create a new bot, and copy the **Bot Token**.
2. **Pinecone**
   - Sign up, create an index named `mysecondmind` (Dimensions: `768`, Metric: `cosine`), and copy your **API Key**.
3. **Google AI Studio**
   - Create a project and get your **API Key** for **Google Gemini Embeddings**.
4. **Groq**
   - Sign up and get your **API Key** for **Llama-3.1-8B** (fast chat).
5. **NVIDIA Build**
   - Sign up and get your **API Key** for **Nemotron-70B** (deep thinking / planning).

---

## 6. Data Flow

### 6.1 Save Memory

1. Input received.
2. LLM classifies and extracts structure.
3. Deadline extraction performed.
4. Embedding generated.
5. Priority score computed.
6. Object stored in database.

---

### 6.2 Query Memory

1. User query embedded.
2. Vector similarity search.
3. LLM reranking.
4. Structured result returned.

---

### 6.3 Night Summary

1. Fetch items created today.
2. Analyze dominant tags.
3. Detect thematic clustering.
4. Generate structured reflection report.

---

### 6.4 Resurfacing Engine

1. Query eligible old memories.
2. Compute resurfacing score.
3. Select top N.
4. Deliver contextualized recall message.

---

## 7. Design Principles

### 7.1 Separation of Concerns

* Inference ≠ Storage
* Storage ≠ Scheduling
* Scheduling ≠ UI

Each module is independently replaceable.

---

### 7.2 Local First

* No external APIs
* No data leakage
* No token dependency
* No rate limits

The user owns cognition.

---

### 7.3 Evolvability

System supports extension toward:

* Knowledge graph modeling
* Attention vector tracking
* Topic clustering
* Research publication
* Multi-agent orchestration

---

## 8. Future Extensions

* Knowledge graph layer (entity linking)
* Attention drift modeling
* Adaptive resurfacing frequency
* Personal cognitive performance metrics
* Focus alignment prediction
* Feedback-aware resurfacing (learning from “relevant / not relevant / snooze / archive” signals)
* User-configurable serenity levels (how proactive cortexa should be)
* Monthly and yearly diary generation (narrative summaries of your tunnels, projects, and focus)
* Longitudinal personal profile building (interests, habits, cognitive trends over years)
* Research-grade memory modeling

---

## 9. Cognitive Dynamics and System Behavior

### 9.1 Forgetting and Pruning

A system that remembers everything becomes noisy. cortexa incorporates a **forgetting mechanism** inspired by synaptic pruning :

* Memories with consistently low activation (rarely accessed, low relevance, negative feedback) decay over time.
* Very low-activation items can be:
  * Archived (moved out of the active recall set).
  * Down-weighted in resurfacing and RAG contexts.
* This keeps tunnels and retrieval results sharp instead of clogged with stale notes.

### 9.2 Sleep Cycle (Offline Processing)

Biological brains consolidate memory during sleep. cortexa mirrors this with **offline processing**:

* A scheduled background job runs when the user is idle:
  * Updates resurfacing scores.
  * Maintains tunnels (cluster updates, growth/dormancy).
  * Applies forgetting/pruning rules.
  * Generates nightly reflection reports and daily digests.

### 9.3 Emotional Valence and Friction

Not all memories are equal. Some are emotionally charged or represent “stuck” work:

* Each memory can carry:
  * **Valence** — rough sentiment (positive/negative/neutral).
  * **Friction** — degree of blockage or effort (e.g., “stuck on this problem”).
* These signals help the system:
  * Prioritize resurfacing of high-friction, high-importance problems.
  * Distinguish exciting breakthroughs from trivial logs.

### 9.4 Cold Start Strategy

On Day 1, the system should already feel useful:

* Optional **passive ingestion** paths:
  * Import past chats (Telegram exports, etc.).
  * Import selected notes or documents.
  * Pull in a limited browser/bookmark history.
* These imports are embedded and indexed to immediately populate tunnels and semantic space.

### 9.5 Serenity and Nagging Threshold

Proactivity is powerful but can become annoying:

* The system maintains a **serenity setting** / nagging threshold:
  * Controls how often digests, resurfacing prompts, and reminders are pushed.
  * Adapts over time based on user behavior (dismissing or engaging with notifications).
* Goal: Be a calm, reliable companion—not a noisy assistant.

---

## 10. Non-Goals

* Replacing full project management tools
* Acting as a general-purpose chatbot
* Serving multi-tenant public users (initially)

This is a personal cognitive engine.

---

## 11. Why This Matters

Cognitive performance determines output.

Most people increase effort.
Very few increase memory structure.

cortexa increases:

* Continuity of thought
* Cross-domain synthesis
* Long-term idea retention
* Strategic awareness

It transforms memory from passive archive into active intelligence.

---

## 12. Vision

At maturity, cortexa becomes:

* A structured mirror of your intellectual evolution.
* A cognitive amplifier.
* A long-term thinking companion.
* A research platform for modeling digital cognition.

---

## 13. P0 Core Reliability Runbook

### Feature flags

- `ACTION_ROUTER=true` enables the structured action router path.
- `DEBUG_MODE=true` prints compact routing/action traces in Telegram.

### Smoke prompts

Run these after deploy/restart:

1. `What can help me with recording my laptop screen`
   - Expect: query answer, not saved as note.
2. `"The Infinity Machine" https://...`
   - Expect: heading preserved as title, link extracted/saved.
3. Re-send the same text or same URL
   - Expect: `You have already saved this before.` + dashboard `Open:` link.
4. `what poem did i save about a girl`
   - Expect: poem-only filtered list with dashboard links.

### Telegram `409 Conflict` (“only one bot instance”)

Long polling (`run_polling`) allows **exactly one** process (worldwide) to call `getUpdates` per bot token.

**Common causes**

This is **not** about how many *people* use the bot—it is about how many *server processes* poll Telegram with the same token.

- **More than one app instance (replica)** on any host (e.g. Koyeb/Railway/Fly/Heroku with **instances > 1** or autoscaling). Fix: set **exactly 1 instance / 1 replica** for the service that runs `main.py`, or move to webhooks.
- **Same token in two deployments** (two Koyeb services, prod + stale preview, or a local `python main.py` while Koyeb is live). Fix: stop the extra deployment or use a **separate bot token** for local dev.

**Quick checks:** In your PaaS dashboard, confirm **one** running instance for this service; search for a second service or old deployment using the same `TELEGRAM_TOKEN`. Brief 409s during a rolling deploy can happen until the old instance stops.

### Regression gate command

From `exocortex/`:

`python scripts/eval_p0_core_gate.py`

Optional override:

`EVAL_CASES_FILE=scripts/eval_cases_p0.json python scripts/eval_p0_core_gate.py`

### Rollback

- Set `ACTION_ROUTER=false` to force legacy intent path.
- Restart bot service.
