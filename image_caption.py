import requests
import base64
import mimetypes
import dotenv
import os

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


dotenv.load_dotenv()
API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
API_KEY_2 = os.getenv("GEMINI_API_KEY_2")


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def get_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"

def generate_image_caption(image_path):
    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64
                        }
                    },
                    {
                        "text": "Describe this image in one sentence."
                    }
                ]
            }
        ]
    }
    for api_key in (API_KEY_1, API_KEY_2):
        response = requests.post(
            f"{API_URL}?key={api_key}",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return "No caption generated."
        elif response.status_code in (401, 403, 429):
            # Try next key on quota/auth/rate limit errors
            continue
        else:
            return f"Error: {response.status_code} - {response.text}"
    return f"Error: All API keys failed. Last response: {response.status_code} - {response.text}"