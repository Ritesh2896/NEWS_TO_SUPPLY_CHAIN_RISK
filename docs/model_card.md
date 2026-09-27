# BDS-35 Model/System Card

Purpose: identify supply-chain disruption events from news and estimate supplier risk.

Inputs: news articles or safely simulated/curated snippets.

Outputs: event, supplier, product, location, exposure, risk score, risk level and alert.

Current approach: keyword relevance/event extraction, lightweight entity extraction, string-similarity entity linking, deduplication, graph relationships and weighted risk scoring.

Limitations: development dataset is small and includes synthetic supply-chain data; event categories are limited; entity linking can produce false matches; risk scoring is a domain-designed heuristic rather than a learned causal model.
