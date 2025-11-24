
import asyncio
import json
import os
from pathlib import Path
from typing import Set

DAEMON_SOCKET_PATH = Path.home() / ".nerion" / "daemon.sock"
event_clients: Set = set()

async def listen_to_daemon_events():
    print("Starting standalone listener...")
    try:
        print(f"Connecting to {DAEMON_SOCKET_PATH}...")
        reader, writer = await asyncio.open_unix_connection(DAEMON_SOCKET_PATH)
        print("Connected!")
        
        while True:
            try:
                print("Waiting for line...")
                line = await reader.readline()
                if not line:
                    print("EOF")
                    break
                
                print(f"Received {len(line)} bytes")
                message = json.loads(line.decode())
                print(f"Decoded message type: {message.get('type')}")
                
                if event_clients:
                    print("Broadcasting...")
                    # Simulate broadcast
                    pass
                    
            except json.JSONDecodeError as e:
                print(f"JSON Error: {e}")
            except Exception as e:
                print(f"Loop Error: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(listen_to_daemon_events())
