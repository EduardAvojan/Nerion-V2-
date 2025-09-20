import os
import json
import pathlib
import datetime
from selfcoder.analysis.search_api import search_enriched, _serpapi_search_raw

os.environ['NERION_SEARCH_PROVIDER'] = 'serpapi'
os.environ['NERION_SEARCH_API_KEY'] = os.getenv('SERPAPI_API_KEY','')

q = "what's the weather tomorrow in los angeles"
enriched = search_enriched(q, n=5, freshness='day')
raw = _serpapi_search_raw(q, n=5, freshness='day', allow=None)

print("=== ENRICHED ===")
print(json.dumps(enriched, indent=2)[:6000])
print("\n=== RAW KEYS ===", list((raw.get("data") or {}).keys()))

out = pathlib.Path("out/debug")
out.mkdir(parents=True, exist_ok=True)
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
(out / f"enriched_{ts}.json").write_text(json.dumps(enriched, indent=2))
(out / f"serpapi_raw_{ts}.json").write_text(json.dumps(raw.get("data", {}), indent=2))
print("Saved to", out)
