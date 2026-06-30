from pathlib import Path

from metarag.corpus import load_corpus, load_seed_qas


def test_corpus_and_seed_bank_are_large_enough_for_mvp():
    chunks = load_corpus(Path("data/nist_ai_rmf_corpus.md"))
    qas = load_seed_qas(Path("data/seed_qa.yaml"))

    assert len(chunks) >= 12
    assert len(qas) == 20
    assert {qa.supporting_section for qa in qas}.issubset({chunk.section for chunk in chunks})
