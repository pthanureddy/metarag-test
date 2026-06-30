from metarag.metamorphic import generate_paraphrases


def test_generate_paraphrases_has_requested_count_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    paraphrases = generate_paraphrases("What is the main purpose of the Govern function?", n=3)

    assert len(paraphrases) == 3
    assert all("?" in item for item in paraphrases)
