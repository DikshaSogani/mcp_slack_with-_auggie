import os
import json
from dotenv import load_dotenv
from slack_sdk import WebClient

load_dotenv()

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def fetch_channels():
    try:
        response = client.conversations_list(types="public_channel")
        channels = response.get("channels", [])
        channel_map = {}
        for c in channels:
            channel_map[c["name"]] = c["id"]
        
        with open("channels.json", "w", encoding="utf-8") as f:
            json.dump(channel_map, f, indent=4)
        print(f"Successfully saved {len(channel_map)} channels to channels.json")
    except Exception as e:
        print(f"Error fetching channels: {e}")

if __name__ == "__main__":
    fetch_channels()