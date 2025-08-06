import requests

payload = {
    "url": "https://emttwzssiuztjcjapurw.supabase.co/storage/v1/object/public/useruploads/images/0772efb8-b210-4e15-8a19-e59107f6b4ad.jpg",
    "latitude": 19.0760,
    "longitude": 72.8777
}

r = requests.post("https://project-vlo5.onrender.com/analyze", json=payload)
print("Status Code:", r.status_code)
try:
    print("Response:", r.json())
except Exception:
    print("Response (raw):", r.text)
