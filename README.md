# multi-modal

FastAPI backend scaffold for a real-time AI-assisted AR workflow.

## Current scope

The project is structured for the first milestone: accept an image over WebSocket, preprocess it, send it to an OpenAI vision model, and stream text back immediately.

The backend now includes a minimal LangGraph agent that:
- keeps short-lived per-client session memory in process
- classifies the request into a mode such as `describe_scene`, `identify_component`, `read_text`, or `analyze_circuit`
- runs the vision model with prior session context
- stores the latest turn for the next image

## Layout

```text
app/
  api/routes/        HTTP and WebSocket endpoints
  agents/            LangGraph state and graph orchestration
  core/              app settings and shared configuration
  schemas/           request and event data models
  services/          vision, image preparation, session memory, rate limiting
  websocket/         connection management and frame protocol handling
```

## Run

```bash
uvicorn app.main:app --reload
```

## Environment

Set provider env vars before using the identify WebSocket.

The app now supports:
- `MODEL_PROVIDER=openai` with `OPENAI_API_KEY` and `OPENAI_MODEL`
- `MODEL_PROVIDER=groq` with `GROQ_API_KEY`, `GROQ_MODEL`, and `GROQ_BASE_URL`

If your current environment does not already have it, install `langgraph` to enable the agent path.
