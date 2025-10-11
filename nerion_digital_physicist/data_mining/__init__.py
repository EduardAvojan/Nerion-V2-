"""Data mining module for extracting quality code examples from public sources.

This module provides tools for scraping and filtering code improvements from
GitHub, preparing them for GNN training.
"""

from nerion_digital_physicist.data_mining.github_api_connector import (
    GitHubAPIConnector,
    build_search_queries,
)
from nerion_digital_physicist.data_mining.github_quality_scraper import (
    CommitData,
    GitHubQualityScraper,
    QualityThresholds,
    ScraperStats,
)

__all__ = [
    "GitHubAPIConnector",
    "GitHubQualityScraper",
    "CommitData",
    "QualityThresholds",
    "ScraperStats",
    "build_search_queries",
]
