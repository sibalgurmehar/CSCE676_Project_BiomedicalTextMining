# BioSeek Project Story

BioSeek is built as a course project, not as a startup product. The goal is to answer a clear research question with a system that is simple enough to understand, defend, and demonstrate.

## Problem framing

Biomedical search is not just about getting relevant documents. Different retrieval methods may surface different regions of the literature, different topic clusters, and different levels of redundancy. A method that looks strong on standard relevance metrics may still produce a narrower or less informative view of the evidence space.

BioSeek studies that tradeoff directly.

## What makes the project academically interesting

- It compares lexical, semantic, and hybrid retrieval over the same biomedical corpus.
- It uses an established benchmark source, BioASQ Task B, for question and relevance supervision.
- It treats diversity and structure as first-class outcomes, not only relevance.
- It creates room for both classic IR evaluation and simple data mining analysis.

## Why this project is scoped the way it is

This project intentionally avoids extra product features. There are no user accounts, collaboration tools, or social features. Those would add implementation effort without strengthening the research argument.

Instead, the project focuses on:

- a unified PMID-based dataset
- understandable retrieval baselines
- measurable evaluation outputs
- lightweight analysis views
- a clean UI for demonstrating results

## Expected demo story

1. Introduce the dataset construction pipeline.
2. Show the four retrieval methods on the same biomedical query.
3. Compare top-ranked PubMed articles and relevance overlap.
4. Show how result diversity and clustering differ by method.
5. Discuss which methods work best for which query types.

## Final takeaway

BioSeek is meant to show that retrieval quality in biomedical IR should be judged not only by relevance, but also by the breadth and structure of the information surfaced.

