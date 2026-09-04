# 🌍 AtlasAI — Multi-Agent Travel Planner 

AtlasAI turns a single sentence like *"Plan a 3-day trip to Tokyo with a budget of $1200"* into a full travel plan — flight options, hotel picks, and a day-by-day schedule — by routing the request through a chain of specialized AI agents built on **LangGraph**.

Instead of one model trying to do everything at once, the request moves through a pipeline: a flight agent gathers live flight data, a hotel agent researches places to stay, an itinerary agent turns that into a realistic schedule, and a final agent writes it all up into one clean, readable answer.

Live: https://atlasai-multi-agent-travel-planner.onrender.com/

---

## How a request flows through the system

```
User message
     │
     ▼
┌─────────────┐
│ Flight Agent│  → queries live flight data via a flight-search tool
└─────┬───────┘
      ▼
┌─────────────┐
│ Hotel Agent │  → runs a Tavily web search for accommodation options
└─────┬───────┘
      ▼
┌────────────────┐
│ Itinerary Agent│ → LLM builds a day-by-day plan from the flight + hotel data
└─────┬──────────┘
      ▼
┌──────────────┐
│ Final Agent  │  → assembles trip summary, budget breakdown & recommendations
└──────────────┘
      │
      ▼
 Polished response back to the user
```

This is implemented as a `StateGraph` in LangGraph, with a shared `TravelState` object (query, flight results, hotel results, itinerary, message history, and an LLM-call counter) passed between nodes. Conversations are checkpointed to **PostgreSQL** via `PostgresSaver`, so a `thread_id` can be reused to continue a planning session instead of starting over.

A guiding rule baked into the agent prompts: **never invent data**. If flight pricing, hotel rates, or availability aren't returned by the tools, the response says so explicitly rather than guessing.

## What's under the hood

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph `StateGraph` (4 sequential nodes) |
| LLM | Groq-hosted `openai/gpt-oss-120b` via `langchain-groq` |
| Flight data | Custom flight-search tool (`tools/flight_tool.py`) |
| Hotel / web research | Tavily search (`tools/tavily_tool.py`) |
| Conversation memory | PostgreSQL, via `langgraph-checkpoint-postgres` |
| API layer | FastAPI |
| Frontend | Jinja2 templates + static HTML/CSS/JS |

## Repository layout

```
AtlasAI-Multi-Agent-Travel-Planner/
├── app.py            # FastAPI app: routes, static files, templates
├── backend.py         # LangGraph graph definition + agent nodes
├── tools/              # flight_tool.py, tavily_tool.py — external data sources
├── templates/          # index.html served at "/"
├── static/             # frontend assets
├── requirements.txt
├── Dockerfile
└── test.py
```

## Getting it running locally

**You'll need:**
- Python 3.10+
- A reachable PostgreSQL database (used for checkpointing agent state)
- API keys: **Groq**, **Tavily**, and your flight-data provider

**1. Clone and install**
```bash
git clone https://github.com/Rahuljoshi1216/AtlasAI-Multi-Agent-Travel-Planner.git
cd AtlasAI-Multi-Agent-Travel-Planner
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql://user:password@host:5432/travel_db
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```
> If your `DATABASE_URL` doesn't already specify `sslmode`, the app appends `sslmode=require` automatically.

**3. Run it**
```bash
python app.py
```
Visit **http://127.0.0.1:8000** for the web UI.

### Or run with Docker
```bash
docker build -t atlasai .
docker run -p 8000:8000 --env-file .env atlasai
```

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Basic liveness check |
| `POST` | `/api/travel` | Submit a travel planning request |

**Example call:**
```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message": "Plan a 3-day trip to Tokyo with a budget of $1200"}'
```

**Response shape:**
```json
{
  "success": true,
  "thread_id": "user_xxxxxxxx",
  "answer": "Full formatted travel plan...",
  "flight_results": "...",
  "hotel_results": "...",
  "itinerary": "...",
  "llm_calls": 3
}
```

Reusing the returned `thread_id` in a follow-up request continues the same planning thread rather than starting fresh.

## Final response format

The final agent structures every answer the same way, so results stay consistent and easy to scan:

1. **Trip Summary**
2. **Flight Information**
3. **Hotel Suggestions**
4. **Day-by-Day Itinerary** (morning / afternoon / evening)
5. **Estimated Budget** (flights, hotel, activities, transport, food — marked "Not available" where pricing is missing)
6. **Final Recommendations**

## Contributing

Issues and PRs are welcome — new tools, better prompts, additional agents, or frontend improvements are all fair game. Fork the repo, branch off `main`, and open a pull request.

## License

Released under the MIT License. See [`LICENSE`](./LICENSE) for details.
