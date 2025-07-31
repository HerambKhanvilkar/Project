import requests

url = "https://project-vlo5.onrender.com/analyze"  # ← Correct route!

payload = {
    "url": "https://emttwzssiuztjcjapurw.supabase.co/storage/v1/object/public/useruploads/videos/video%20(1).mp4",
    "latitude": 19.0760,
    "longitude": 72.8777
}

res = requests.post(url, json=payload)

print("Status Code:", res.status_code)
print("Response Text:", res.text)
