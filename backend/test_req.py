import urllib.request
import json
url = 'http://localhost:5000/api/chat'
headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
data = {'message': 'hello', 'context': 'Stress'}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
try:
    response = urllib.request.urlopen(req)
    print(response.read().decode('utf-8'))
except Exception as e:
    print(e.read().decode('utf-8'))
