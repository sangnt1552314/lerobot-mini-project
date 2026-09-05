python server.py > server.log 2>&1 &
~/bin/ngrok http 8000 --url https://default.internal

# ps aux | grep -E "server.py|uvicorn" | grep -v grep