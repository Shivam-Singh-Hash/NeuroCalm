import os
import urllib.request
import json
api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/........"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
payload = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": "hello"}]}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
try:
    with urllib.request.urlopen(req) as res:
        print(res.read().decode())
except Exception as e:
    print(e.read().decode())
