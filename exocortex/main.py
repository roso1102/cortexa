import threading

from dotenv import load_dotenv
load_dotenv()  # load .env before anything else reads env vars

from flask import Flask

from src.config import load_config, ConfigError
from src.memory import MemoryManager
from src.reflection import ReflectionService
from src.telegram_bot import run_bot
from src.api import create_api_blueprint
from src.db import init_db


app = Flask(__name__)


@app.get("/")
def health() -> str:
    return "Alive", 200


def run_flask() -> None:
    app.run(host="0.0.0.0", port=8080)


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    # Option B: canonical Postgres foundation (best-effort init).
    # If DATABASE_URL is not configured yet, this is a no-op.
    init_db()

    # Dashboard API layer — always registered; auth handled inside the blueprint.
    # If DASHBOARD_SECRET is not set, every /api/* request returns 401.
    memory = MemoryManager(config)
    reflection = ReflectionService(config, memory)
    api_bp = create_api_blueprint(
        memory=memory,
        reflection=reflection,
        dashboard_secret=config.dashboard_secret,
        dashboard_users=config.dashboard_users,
        dashboard_public_url=config.dashboard_public_url,
    )
    app.register_blueprint(api_bp)

    # Flask runs in a background daemon thread so the main thread is free for the bot
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Telegram bot runs in the main thread — it creates its own asyncio event loop
    run_bot(config)


if __name__ == "__main__":
    main()
