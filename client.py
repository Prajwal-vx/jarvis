"""OpenAI client interface and smoke test for Jarvis with automatic .env loading."""

import os
from pathlib import Path


def load_env_file(dotenv_path=None):
    """Load key-value pairs from a .env file into os.environ if not already set."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent / ".env"
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.is_file():
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as err:
        print(f"Notice: Failed to parse .env file: {err}")


# Load environment variables on module import
load_env_file()


def is_openai_configured():
    """Return True if OPENAI_API_KEY is available in the environment."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def get_openai_client():
    """Return an initialized OpenAI client or None if unconfigured or missing library."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as err:
        print(f"Warning: Could not initialize OpenAI client: {err}")
        return None


def ask_openai(prompt, system_prompt="You are a concise virtual assistant named Jarvis. Keep answers short and conversational."):
    """Send a prompt to OpenAI and return the text response with safe error handling."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the OpenAI query.")

    try:
        from openai import OpenAI, OpenAIError
    except ImportError as error:
        raise ImportError(
            "The 'openai' package is required. Install it with: python -m pip install openai"
        ) from error

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            timeout=15.0,
        )
        return completion.choices[0].message.content.strip()
    except OpenAIError as oerr:
        return f"OpenAI service error: {oerr}"
    except Exception as exc:
        return f"Failed to get AI response: {exc}"


def ask_openai_with_context(history, system_prompt="You are a concise virtual assistant named Jarvis. Keep answers short and conversational."):
    """Send conversation history to OpenAI and return the latest response."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Set OPENAI_API_KEY before running the OpenAI query.")

    try:
        client = get_openai_client()
        if client is None:
            return "Failed to initialize the OpenAI client."
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}, *history],
            timeout=15.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        return f"Failed to get AI response: {exc}"


if __name__ == "__main__":
    if is_openai_configured():
        print(ask_openai("what is coding"))
    else:
        print("OPENAI_API_KEY is not set in environment or .env file.")
