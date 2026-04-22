import redis
import json

r = redis.from_url("redis://localhost:6379/0", decode_responses=True)

print("=== Incoming Stream ===")
entries = r.xrange("news:incoming", "-", "+")
for msg_id, fields in entries:
    state = json.loads(fields["data"])
    print(f"  {msg_id} | {state}")