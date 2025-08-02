import requests
payload = {
  "url": "https://emttwzssiuztjcjapurw.supabase.co/storage/v1/object/public/useruploads/videos/video%20(1).mp4",
  "latitude": 19.0760,
  "longitude": 72.8777
}
r = requests.post("https://project-vlo5.onrender.com/analyze", json=payload)
print(r.status_code, r.json())