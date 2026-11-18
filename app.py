"""Entry point kept for compatibility: ``uvicorn app:app``."""

from ovp.api.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ovp.api.app:app", host="0.0.0.0", port=8000)
