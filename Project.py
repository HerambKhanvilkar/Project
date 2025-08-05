import requests

r = requests.get("https://project-vlo5.onrender.com/")
print("Status Code:", r.status_code)
print("Response:", r.text)
