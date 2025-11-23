
import asyncio
import json
import os
from pathlib import Path

DAEMON_SOCKET_PATH = Path.home() / ".nerion" / "daemon.sock"

async def listen():
    print(f"Connecting to {DAEMON_SOCKET_PATH}...")
    try:
        reader, writer = await asyncio.open_unix_connection(DAEMON_SOCKET_PATH)
        print("Connected! Waiting for data...")
        
        while True:
            line = await reader.readline()
            if not line:
                print("Connection closed by daemon.")
                break
            
            print(f"Received: {line.decode().strip()}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nStopped.")
