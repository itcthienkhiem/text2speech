import requests

API_URL = "https://api-inference.huggingface.co/models/nguyenvulebinh/wav2vec2-base-vietnamese-250h"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query(r"C:\Users\khiem.nthien\Downloads\audio(1).wav")
print(output)
