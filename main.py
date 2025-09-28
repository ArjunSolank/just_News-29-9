#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Important News App — Flask API-enabled version
Features:
- Rich console output
- Sound notifications
- City-aware news + keyword prefilter
- HuggingFace optional classification
- Deduplication
- Flask API endpoints for Flutter frontend
"""

import os
import time
import signal
import requests
import feedparser
import threading
import webbrowser
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv

# Optional sound
try:
    from playsound import playsound
except Exception:
    playsound = None

# -------------------- Flask imports --------------------
from flask import Flask, jsonify, request  # For API server
from flask_cors import CORS  # To allow Flutter app requests

# -------------------- Load env --------------------
load_dotenv()
console = Console()

# -------------------- Config --------------------
RSS_URL = os.getenv("RSS_URL", "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en")
USER_CITY = os.getenv("USER_CITY", "Delhi").strip()
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "180"))
SHOW_EVERY_CYCLE_SUMMARY = os.getenv("SHOW_EVERY_CYCLE_SUMMARY", "1") == "1"

HF_ENABLE = os.getenv("HF_ENABLE", "1") == "1"
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/facebook/bart-large-mnli")
HF_SCORE_THRESHOLD = float(os.getenv("HF_SCORE_THRESHOLD", "0.50"))
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "12"))
HF_MAX_PER_CYCLE = int(os.getenv("HF_MAX_PER_CYCLE", "20"))

SOUND_FILE = os.getenv("SOUND_FILE", "").strip()
SOUND_ENABLE = os.getenv("SOUND_ENABLE", "1") == "1"
SEEN_CAPACITY = int(os.getenv("SEEN_CAPACITY", "1000"))

CITY_SYNONYMS: Dict[str, List[str]] = {
    "delhi": ["delhi", "new delhi", "ndl", "ncr", "दिल्ली", "नई दिल्ली", "dilli"],
    "mumbai": ["mumbai", "bombay", "मुंबई"],
    "bengaluru": ["bengaluru", "bangalore", "बेंगलुरु"],
}

IMPORTANT_KEYWORDS = [
    "earthquake", "war", "atomic", "nuclear", "blast", "explosion",
    "terrorist", "attack", "murder", "kill", "homicide", "riot", "violence",
    "fire", "flood", "storm", "hurricane", "tornado", "landslide", "accident",
    "robbery", "theft", "cyber attack", "bomb", "hostage", "shooting",
    "curfew", "evacuation", "collapse", "outbreak", "disease", "pandemic"
]

# -------------------- Utilities --------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp string

def normalize_text(s: str) -> str:
    return (s or "").strip().lower()  # lowercase + trim

def city_aliases(city: str) -> List[str]:
    c = normalize_text(city)
    return CITY_SYNONYMS.get(c, [c])

def title_matches_city(title: str, city: str) -> bool:
    t = normalize_text(title)
    return any(alias in t for alias in city_aliases(city))

def keyword_prefilter(title: str, keywords: List[str]) -> bool:
    t = normalize_text(title)
    return any(k in t for k in keywords)

def sound_notify() -> None:
    if not SOUND_ENABLE:
        return
    try:
        if playsound and SOUND_FILE and os.path.exists(SOUND_FILE):
            playsound(SOUND_FILE)
        else:
            console.print("\a", end="")  # fallback beep
    except Exception as e:
        console.print(f"[red]❌ Sound error:[/red] {e}")

def hf_classify(text: str) -> Tuple[Optional[str], float]:
    if not HF_ENABLE or not HF_API_KEY:
        return None, 0.0
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": text, "parameters": {"candidate_labels": IMPORTANT_KEYWORDS}}
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=HF_TIMEOUT)
        data = resp.json()
        if "labels" in data and "scores" in data:
            return data["labels"][0], float(data["scores"][0])
        return None, 0.0
    except Exception:
        return None, 0.0

def fetch_rss(url: str) -> List[Dict[str, Any]]:
    try:
        feed = feedparser.parse(url)
        items = [{"title": e.title, "link": e.link} for e in getattr(feed, "entries", [])]
        return items
    except Exception as e:
        console.print(f"[red]RSS fetch error:[/red] {e}")
        return []

class BoundedDedup:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.queue = deque(maxlen=capacity)
        self.set = set()
    def seen(self, key: str) -> bool:
        k = key.strip()
        if k in self.set:
            return True
        self.queue.append(k)
        self.set.add(k)
        while len(self.set) > len(self.queue):
            for s in list(self.set):
                if s not in self.queue:
                    self.set.remove(s)
                    break
            else:
                break
        return False

# -------------------- Flask API Setup --------------------
app = Flask(__name__)  # Create Flask app
CORS(app)  # Allow all origins (Flutter can call)

# Global news cache
latest_news = []
important_news = []

@app.route("/")
def root():
    return jsonify({"msg": "✅ News Now API running!", "version": "1.0.0"})

@app.route("/news")
def api_news():
    return jsonify({"news": latest_news})  # Return latest news list

@app.route("/important")
def api_important():
    return jsonify({"important": important_news})  # Return filtered important news

@app.route("/city", methods=["GET", "POST"])
def api_city():
    global USER_CITY
    if request.method == "POST":
        data = request.json
        if "city" in data:
            USER_CITY = data["city"]
            return jsonify({"status": "success", "city": USER_CITY})
        return jsonify({"status": "error", "message": "City not provided"})
    return jsonify({"city": USER_CITY})

# -------------------- Main loop --------------------
running = True
def handle_sigint(signum, frame):
    global running
    running = False
signal.signal(signal.SIGINT, handle_sigint)

def print_header():
    table = Table.grid(expand=True)
    table.add_column(justify="center")
    table.add_row(f"[bold cyan]🌍 Live Important News App[/bold cyan]")
    table.add_row(f"[dim]RSS:[/dim] {RSS_URL}")
    table.add_row(f"[dim]City:[/dim] {USER_CITY}  |  [dim]Refresh:[/dim] {REFRESH_INTERVAL}s")
    console.print(Panel.fit(table, border_style="cyan", box=box.ROUNDED))

def update_news():
    """Main news fetching loop"""
    global latest_news, important_news
    dedup = BoundedDedup(SEEN_CAPACITY)
    cycle = 0
    while running:
        cycle += 1
        items = fetch_rss(RSS_URL)
        if not items:
            time.sleep(REFRESH_INTERVAL)
            continue
        cycle_news = []
        cycle_important = []
        hf_budget = HF_MAX_PER_CYCLE
        for news in items:
            title = news.get("title", "")
            link = news.get("link", "")
            if not title or dedup.seen(title):
                continue
            is_city = title_matches_city(title, USER_CITY)
            is_keyword = keyword_prefilter(title, IMPORTANT_KEYWORDS)
            label, score = None, 0.0
            if is_city:
                label, score = "city-priority", 1.0
            elif is_keyword:
                label, score = "keyword", 0.75
            elif HF_ENABLE and HF_API_KEY and hf_budget > 0:
                label, score = hf_classify(title)
                hf_budget -= 1
            news_item = {
                "title": title,
                "link": link,
                "category": label or "general",
                "score": float(score),
                "time": now_str(),
                "is_important": (label == "city-priority") or (score >= HF_SCORE_THRESHOLD) or (is_keyword and score >= 0.0)
            }
            cycle_news.append(news_item)
            if news_item["is_important"]:
                cycle_important.append(news_item)
                sound_notify()
                console.print(Panel.fit(
                    f"[bold yellow]{title}[/bold yellow]\n\n[blue]🔗 {link}[/blue]\n\n[green]🕒 {now_str()}[/green]",
                    title="🔥 Important News",
                    subtitle=f"Category: {label or '—'} | Score: {score:.2f}",
                    border_style="red"
                ))
        latest_news = cycle_news
        important_news = cycle_important
        if SHOW_EVERY_CYCLE_SUMMARY:
            console.print(f"[dim]{now_str()} — Cycle {cycle} complete. Important: {len(cycle_important)} | Total: {len(items)}[/dim]")
        time.sleep(REFRESH_INTERVAL)

# -------------------- Run both Flask + News loop --------------------
if __name__ == "__main__":
    print_header()
    # Start news loop in background thread
    news_thread = threading.Thread(target=update_news, daemon=True)
    news_thread.start()

    # Detect Render/Heroku/Production (uses PORT from env)
    port = int(os.environ.get("PORT", 5000))

    try:
        # Use waitress in production
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
    except ImportError:
        # Fallback to Flask dev server (local run)
        app.run(host="0.0.0.0", port=port)
