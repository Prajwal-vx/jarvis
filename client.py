"""Optional OpenAI smoke test with no import-time side effects."""

import os


def ask_openai(prompt):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError(
            "The 'openai' package is required. Install it with: "
            "python -m pip install openai"
        ) from error

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the OpenAI smoke test.")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a concise virtual assistant named Jarvis."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    print(ask_openai("what is coding"))