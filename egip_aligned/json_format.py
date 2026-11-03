import json
import re

def parse(raw):
    try:
        value=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
        return value if isinstance(value,dict) else None
    except (ValueError,TypeError): return None
