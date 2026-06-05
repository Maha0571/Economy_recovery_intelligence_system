import requests


def chat_with_ollama(prompt: str) -> str:
    """
    Send prompt to local Ollama LLM and return response text.
    """

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # check HTTP errors

        result = response.json()
        return result.get("response", "No response from model")

    except requests.exceptions.ConnectionError:
        return " Ollama server not running. Run: ollama serve"

    except Exception as e:
        return f" Error: {str(e)}"