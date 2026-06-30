# MetaRAG-Test Report

## Summary

| Metric | Value |
|---|---:|
| groundedness accuracy | 1.000 |
| hallucination catch rate | 1.000 |
| metamorphic violation rate | 0.000 |
| mean flakiness variance | 0.000 |
| ragas agreement | 1.000 |

## Dataset

The corpus is a compact public NIST AI RMF study corpus, chosen to avoid employer IP and to focus the project on testing RAG behavior.

## Validation

The runner validates groundedness on clean retrieved context, then deliberately corrupts retrieved context and checks whether the oracle catches unsupported answers.