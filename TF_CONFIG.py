import json
import os

tf_config = {"cluster":{"worker": ["10.132.0.9:2222", "10.132.0.10:2222", "10.132.0.11:2222"]},"task": {"type": "worker", "index": 1}}

print('An example TF_CONFIG')
print(json.dumps(tf_config))
