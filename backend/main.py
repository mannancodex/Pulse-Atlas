
"""Pulse Atlas API: live categorised news, topic trends, and summaries."""

import re
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

ENT = re.compile(
    r"[A-Z][A-Za-z0-9'’&.-]+(?:\s[A-Z][A-Za-z0-9'’&.-]+){0,2}"
)

CACHE = {}
LOCK = threading.RLock()
TTL = 600

app = FastAPI(title="Pulse Atlas API")

# Allow your GitHub Pages frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mannancodex.github.io"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def parse_feed(url):
    response = requests.get(url, headers=UA, timeout=12)
    response.raise_for_status()

    feed = feedparser.parse(response.content)
    out = []
    seen = set()

    for entry in feed.entries:
        title = re.sub(r"\s+-\s+[^-]+$", "", entry.get("title", "")).strip()

        if (
            not title
            or title.lower() in seen
            or not entry.get("published_parsed")
        ):
            continue

        seen.add(title.lower())
        source = entry.get("source") or {}

        out.append({
            "title": title,
            "link": entry.get("link", ""),
            "source": source.get("title", ""),
            "source_url": source.get("href", ""),
            "dt": datetime(
                *entry.published_parsed[:6],
                tzinfo=timezone.utc,
            ),
        })

    return sorted(out, key=lambda item: item["dt"], reverse=True)


def fetch_cat(cat, places, cc):
    """Search Google News, widening the location when necessary."""
    items = []
    used = places[0]

    for place in places:
        query = quote_plus(f'({CATS[cat]}) "{place}" when:7d')

        urls = [
            f"https://news.google.com/rss/search?q={query}"
            f"&hl=en-{cc}&gl={cc}&ceid={cc}:en",
            f"https://news.google.com/rss/search?q={query}"
            "&hl=en-US&gl=US&ceid=US:en",
        ]

        for url in urls:
            try:
                items = parse_feed(url)
            except Exception:
                items = []

            if items:
                break

        used = place

        if len(items) >= 10:
            break

    return cat, items, used


def entities(title, exclude):
    found = []

    for match in ENT.findall(title):
        words = match.split()

        while words and words[0].lower() in STOP:
            words = words[1:]

        topic = " ".join(words)

        if len(topic) >= 3 and topic.lower() not in STOP:
            if topic.lower() not in exclude:
                found.append(topic)

    return found


def topics(items, exclude, top=6):
    """Calculate topic shares and percentage-point changes."""
    now = datetime.now(timezone.utc)

    def age(item):
        return now - item["dt"]

    recent_24h = sum(
        age(item) <= timedelta(hours=24) for item in items
    )

    window = 24 if recent_24h >= 8 else 72

    current = [
        item for item in items
        if age(item) <= timedelta(hours=window)
    ]

    previous = [
        item for item in items
        if timedelta(hours=window) < age(item)
        <= timedelta(hours=2 * window)
    ]

    def count_topics(items_list):
        return Counter(
            topic
            for item in items_list
            for topic in set(entities(item["title"], exclude))
        )

    current_counts = count_topics(current)
    previous_counts = count_topics(previous)

    result = []

    for topic, count in current_counts.most_common(25):
        if count < 2:
            break

        share = round(count / len(current) * 100)

        delta = (
            round(
                share
                - previous_counts[topic] / len(previous) * 100
            )
            if previous
            else None
        )

        result.append({
            "t": topic,
            "n": count,
            "share": share,
            "delta": delta,
            "new": bool(previous) and previous_counts[topic] == 0,
        })

    return {
        "window": window,
        "base": len(current),
        "list": result[:top],
    }


_pipe = None


def summarize(titles):
    """Generate a transformer summary, falling back to the first headline."""
    global _pipe

    if _pipe is None:
        try:
            from transformers import pipeline

            _pipe = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
            )
        except Exception:
            _pipe = False

    text = ". ".join(titles[:15])

    if _pipe and text:
        try:
            return _pipe(
                text[:3000],
                max_length=70,
                min_length=25,
                do_sample=False,
                truncation=True,
            )[0]["summary_text"]
        except Exception:
            pass

    return titles[0] if titles else "No recent headlines are available."


def pack(item):
    return {
        "title": item["title"],
        "link": item["link"],
        "source": item["source"],
        "source_url": item["source_url"],
        "published": item["dt"].isoformat(),
    }


# FIX 1: Use an API path, not the full Render URL.
@app.get("/api/news")
def news(
    name: str,
    country: str = "",
    cc: str = "US",
    state: str = "",
    scope: str = "city",
):
    if scope not in ("city", "state", "country"):
        raise HTTPException(status_code=400, detail="Invalid scope.")

    name = name.replace('"', "").strip()
    country = country.replace('"', "").strip()
    state = state.replace('"', "").strip()

    cc = cc.upper() if re.fullmatch(r"[A-Za-z]{2}", cc or "") else "US"

    key = f"{scope}:{name}:{cc}"

    with LOCK:
        cached = CACHE.get(key)

        if cached and time.time() - cached["t"] < TTL:
            return cached["data"]

    if scope == "city":
        places = [name, state, country]
    elif scope == "state":
        places = [name, country]
    else:
        places = [name]

    places = list(dict.fromkeys(place for place in places if place))

    if not places:
        raise HTTPException(status_code=400, detail="A location name is required.")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda category: fetch_cat(category, places, cc),
                CATS,
            )
        )

    excluded = {place.lower() for place in places}

    all_items = {
        item["title"]: item
        for _, items, _ in results
        for item in items
    }

    merged = sorted(
        all_items.values(),
        key=lambda item: item["dt"],
        reverse=True,
    )

    categories = [
        {
            "name": category,
            "widened": used if used != name else "",
            "topics": topics(items, excluded),
            "items": [pack(item) for item in items[:10]],
        }
        for category, items, used in results
    ]

    data = {
        "key": key,
        "place": name,
        "country": country,
        "scope": scope,
        "total": len(merged),
        "top": [pack(item) for item in merged[:12]],
        "topics": topics(merged, excluded, 8),
        "categories": categories,
    }

    titles = {
        category: [item["title"] for item in items]
        for category, items, _ in results
    }
    titles["Top"] = [item["title"] for item in merged]

    with LOCK:
        CACHE[key] = {
            "t": time.time(),
            "data": data,
            "titles": titles,
            "sums": {},
        }

    return data


# FIX 2: Give summaries their own API path.
@app.get("/api/summary")
def summary(key: str, cat: str):
    with LOCK:
        cached = CACHE.get(key)

        if not cached or cat not in cached["titles"]:
            raise HTTPException(
                status_code=404,
                detail="Unknown region; reselect it on the map.",
            )

        if cat in cached["sums"]:
            return {"summary": cached["sums"][cat]}

        titles = cached["titles"][cat]

    generated_summary = summarize(titles)

    with LOCK:
        # Cache may have been refreshed while summarization ran.
        current = CACHE.get(key)

        if current and cat in current["titles"]:
            current["sums"][cat] = generated_summary

    return {"summary": generated_summary}


# Keep this last so API routes are registered before the static frontend.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )
