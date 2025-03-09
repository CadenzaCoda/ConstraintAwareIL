import requests
import os
import json
from pathlib import Path


def ntfy(data: str, *, title=None, priority=3, tags=tuple(), timing=None, server=None, topic=None):
    with open(Path(__file__).resolve().parent / "config/ntfy.json", "r") as f:
        config = json.load(f)
    server = server or config['default_server']
    topic = topic or config['default_topic']
    assert isinstance(priority, int) and 1 <= priority <= 5
    assert isinstance(tags, tuple)  # and all(tag in emoji_list for tag in tags)
    requests.post(os.path.join(server, topic),
                  data=data.encode('utf-8'),
                  headers={
                      "Title": title,
                      "Priority": str(priority),
                      "Tags": ','.join(tags),
                      "Markdown": "yes",
                  })


if __name__ == "__main__":
    ntfy(data="test", title='Python dict send test', tags=('tada', ), topic='research', priority=4)
