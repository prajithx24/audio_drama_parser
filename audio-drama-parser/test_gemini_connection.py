import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError


def main() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_file)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print("Error: GEMINI_API_KEY is not set in the .env file.")
        return

    try:
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input="Explain what an audio drama is in two sentences.",
        )
        print(interaction.output_text)
    except APIError as error:
        print(f"Gemini API error: {error}")
    except (OSError, ValueError) as error:
        print(f"Unable to connect to Gemini: {error}")


if __name__ == "__main__":
    main()