from __future__ import annotations

import pytest

from selfcoder.analysis.domain import classify_query


def test_finance_classification():
    label, conf = classify_query("highest gaining stock today")
    assert label == "finance"
    assert conf >= 0.7


def test_real_estate_classification():
    label, conf = classify_query("where did house prices rise in Los Angeles")
    assert label == "real_estate"
    assert conf >= 0.7


def test_healthcare_classification():
    label, conf = classify_query("what is the most recent drug for alzheimer's")
    assert label == "healthcare"
    assert conf >= 0.7


def test_world_news_classification():
    label, conf = classify_query("today's breaking news")
    assert label == "world_news"
    assert conf >= 0.6


def test_tech_news_classification():
    label, conf = classify_query("breaking tech headlines today")
    assert label in {"tech_news", "world_news"}
    assert conf >= 0.6


def test_site_overview_from_url_only():
    label, conf = classify_query("", url="https://apple.com/")
    assert label == "site_overview"
    assert conf >= 0.7


def test_site_query_with_url_and_task_words():
    label, conf = classify_query("best laptop this month", url="https://hp.com/")
    assert label == "site_query"
    assert conf >= 0.8


def test_general_topic_fallback():
    label, conf = classify_query("summarize climate policy impacts")
    assert label == "general_topic"
    assert conf >= 0.5


def test_real_estate_vs_finance_ambiguity_zip():
    # Mixed finance-ish and housing words should route to real_estate here
    label, conf = classify_query("mortgage rates vs rent in zip 90001")
    assert label == "real_estate"
    assert conf >= 0.7


def test_finance_over_real_estate_keywords():
    # Ensure finance wins when finance tokens dominate
    label, conf = classify_query("stock prices in real estate ETFs")
    assert label == "finance"
    assert conf >= 0.7


def test_hints_boost_real_estate_over_prices():
    # When hints are provided, they should boost the target domain
    hints = {"real_estate": ["prices", "zip", "la", "los angeles"]}
    label, conf = classify_query("prices in LA", hints=hints)
    assert label == "real_estate"
    # With 2+ matches (prices + LA), hinted confidence should be >= 0.82
    assert conf >= 0.82