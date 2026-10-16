"""Pulse Atlas API: hover/click a region -> live categorised news, meaningful trend shares, transformer summaries."""
import re, time, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser, requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

UA = {"User-Agent": "pulse-atlas-demo/1.0 (student project)"}
CATS = {
    "Politics": "politics OR government OR election",
    "Business": "business OR economy OR market",
    "Technology": "technology OR startup OR AI",
    "Sports": "sports OR match OR league",
    "Environment": "environment OR weather OR climate",
    "Health": "health OR hospital OR disease",
    "Entertainment": "entertainment OR film OR music",
    "Science": "science OR research OR space",
}
STOP = set("""the a an of in on to for and or with at by from as is are was were be this that it its after over new says
say more amid into about up out how what why who will has have not no vs us can may could would than their his her you
your our one two first last year years day days today live news here there when where while during before under
january february march april may june july august september october november december monday tuesday wednesday thursday
friday saturday sunday""".split())
ENT = re.compile(r"[A-Z][A-Za-z0-9'’&.-]+(?:\s[A-Z][A-Za-z0-9'’&.-]+){0,2}")
CACHE, LOCK, TTL = {}, threading.Lock(), 600
app = FastAPI(title="Pulse Atlas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mannancodex.github.io",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_feed(url):
    feed = feedparser.parse(requests.get(url, headers=UA, timeout=12).content)
    out, seen = [], set()
    for e in feed.entries:
        title = re.sub(r"\s+-\s+[^-]+$", "", e.get("title", "")).strip()
        if not title or title.lower() in seen or not e.get("published_parsed"):
            continue
        seen.add(title.lower())
        src = e.get("source") or {}
        out.append({"title": title, "link": e.link, "source": src.get("title", ""), "source_url": src.get("href", ""),
                    "dt": datetime(*e.published_parsed[:6], tzinfo=timezone.utc)})
    return sorted(out, key=lambda x: x["dt"], reverse=True)


def fetch_cat(cat, places, cc):
    """Google News RSS search; widen city -> state -> country until >= 10 stories."""
    items, used = [], places[0]
    for p in places:
        q = quote_plus(f'({CATS[cat]}) "{p}" when:7d')
        for loc in (f"hl=en-{cc}&gl={cc}&ceid={cc}:en", "hl=en-US&gl=US&ceid=US:en"):
            try:
                items = parse_feed(f"https://news.google.com/rss/search?q={q}&{loc}")
            except Exception:
                items = []
            if items:
                break
        used = p
        if len(items) >= 10:
            break
    return cat, items, used


def entities(title, exclude):
    out = []
    for m in ENT.findall(title):
        w = m.split()
        while w and w[0].lower() in STOP:
            w = w[1:]
        t = " ".join(w)
        if len(t) >= 3 and t.lower() not in STOP and t.lower() not in exclude:
            out.append(t)
    return out


def topics(items, exclude, top=6):
    """Share = % of stories in the current window that mention the topic.
    Change = percentage-point difference vs the previous window of equal length (24h, or 72h if the feed is sparse)."""
    now = datetime.now(timezone.utc)
    age = lambda i: now - i["dt"]
    W = 24 if sum(1 for i in items if age(i) <= timedelta(hours=24)) >= 8 else 72
    cur = [i for i in items if age(i) <= timedelta(hours=W)]
    prev = [i for i in items if timedelta(hours=W) < age(i) <= timedelta(hours=2 * W)]
    count = lambda arr: Counter(t for i in arr for t in set(entities(i["title"], exclude)))
    cc, pc = count(cur), count(prev)
    out = []
    for t, n in cc.most_common(25):
        if n < 2:
            break
        share = round(n / len(cur) * 100)
        delta = round(share - pc[t] / len(prev) * 100) if prev else None
        out.append({"t": t, "n": n, "share": share, "delta": delta, "new": bool(prev) and pc[t] == 0})
    return {"window": W, "base": len(cur), "list": out[:top]}


_pipe = None
def summarize(titles):
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline
            _pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            _pipe = False
    text = ". ".join(titles[:15])
    if _pipe:
        try:
            return _pipe(text[:3000], max_length=70, min_length=25, do_sample=False, truncation=True)[0]["summary_text"]
        except Exception:
            pass
    return titles[0] if titles else ""


def pack(i):
    return {"title": i["title"], "link": i["link"], "source": i["source"], "source_url": i["source_url"],
            "published": i["dt"].isoformat()}


@app.get("/api/news")
def news(name: str, country: str = "", cc: str = "US", state: str = "", scope: str = "city"):
    if scope not in ("city", "state", "country"):
        raise HTTPException(400, "bad scope")
    name = name.replace('"', "")
    key = f"{scope}:{name}:{cc}"
    hit = CACHE.get(key)
    if hit and time.time() - hit["t"] < TTL:
        return hit["data"]
    places = [name, state, country] if scope == "city" else [name, country] if scope == "state" else [name]
    places = list(dict.fromkeys(p.replace('"', "") for p in places if p))
    cc = cc if re.fullmatch(r"[A-Za-z]{2}", cc or "") else "US"
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda c: fetch_cat(c, places, cc.upper()), CATS))
    exclude = {p.lower() for p in places}
    allitems = {i["title"]: i for _, items, _ in results for i in items}
    merged = sorted(allitems.values(), key=lambda x: x["dt"], reverse=True)
    cats = [{"name": cat, "widened": used if used != name else "", "topics": topics(items, exclude),
             "items": [pack(i) for i in items[:10]]} for cat, items, used in results]
    data = {"key": key, "place": name, "country": country, "scope": scope, "total": len(merged),
            "top": [pack(i) for i in merged[:12]], "topics": topics(merged, exclude, 8), "categories": cats}
    titles = {cat: [i["title"] for i in items] for cat, items, _ in results}
    titles["Top"] = [i["title"] for i in merged]
    CACHE[key] = {"t": time.time(), "data": data, "titles": titles, "sums": {}}
    return data


@app.get("/api/summary")
def summary(key: str, cat: str):
    hit = CACHE.get(key)
    if not hit or cat not in hit["titles"]:
        raise HTTPException(404, "Unknown region; reselect it on the map.")
    if cat not in hit["sums"]:
        with LOCK:
            hit["sums"][cat] = summarize(hit["titles"][cat])
    return {"summary": hit["sums"][cat]}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True))
