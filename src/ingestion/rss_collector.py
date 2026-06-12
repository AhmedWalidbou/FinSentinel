import feedparser
import json
import os
from datetime import datetime
from typing import List


RSS_FEEDS = {
    "les_echos": "https://feeds.lesechos.fr/rss/rss_finance.xml",
    "bfm_business": "https://www.bfmtv.com/rss/economie/",
    "reuters_fr": "https://fr.reuters.com/rssFeed/businessNews",
    "boursorama": "https://www.boursorama.com/rss/actualites/"
}

OUTPUT_FILE = "data/raw/articles_raw.json"


def fetch_feed(name: str, url: str) -> List[dict]:
    """
    Fetch articles from a single RSS feed.
    Returns list of article dicts with metadata.
    """
    print(f"Fetching: {name}")
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries:
        article = {
            "source": name,
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "collected_at": datetime.now().isoformat()
        }
        if len(article["summary"]) > 50:
            articles.append(article)

    print(f"  → {len(articles)} articles collected from {name}")
    return articles


def collect_all_feeds() -> List[dict]:
    """
    Collect articles from all RSS feeds.
    Saves to JSON file.
    """
    all_articles = []

    for name, url in RSS_FEEDS.items():
        try:
            articles = fetch_feed(name, url)
            all_articles.extend(articles)
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    os.makedirs("data/raw", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\nTotal articles collected: {len(all_articles)}")
    print(f"Saved to: {OUTPUT_FILE}")
    return all_articles


if __name__ == "__main__":
    articles = collect_all_feeds()
    if articles:
        print(f"\nSample article:")
        print(f"  Source : {articles[0]['source']}")
        print(f"  Title  : {articles[0]['title']}")
        print(f"  Summary: {articles[0]['summary'][:150]}...")