import requests

BASE_URL = "http://127.0.0.1:8000"  # adjust if your app runs elsewhere

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print("Health check:", resp.status_code)
    print(resp.json())

def test_translation():
    # English -> Nepali
    payload_en_np = {"text": "Hello, how are you?", "direction": "en_np"}
    resp = requests.post(f"{BASE_URL}/translate", json=payload_en_np)
    print("\nEnglish -> Nepali:")
    print(resp.status_code, resp.json())

    # # Nepali -> English
    # payload_np_en = {"text": "तिमीलाई कस्तो छ?", "direction": "np_en"}
    # resp = requests.post(f"{BASE_URL}/translate", json=payload_np_en)
    # print("\nNepali -> English:")
    # print(resp.status_code, resp.json())


if __name__ == "__main__":
    test_health()
    test_translation()
