import requests
import json

response = requests.post(
    "http://127.0.0.1:8000/generate/stream",
    json={
        "question": "What is the capital of France?",
        "max_gen": 40
    },
    stream=True
)

response.raise_for_status()

print("Streaming response:\n")

for line in response.iter_lines(decode_unicode=True):

    if line:

        if line.startswith("data: "):

            data = line[6:]

            if data == "[DONE]":
                print("\n\nGeneration completed.")
                break

            result = json.loads(data)

            if "token" in result:
                print(result["token"], end="", flush=True)

            elif "error" in result:
                print("\nError:", result["error"])
                break

response.close()