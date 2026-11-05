from grounding_audit import build_grounding_ledger, split_answer_sentences


def test_all_sentences_with_citations_are_complete():
    answer = "The system processed 1200 records. The report was published in 2025."
    sentences = split_answer_sentences(answer)
    ledger = build_grounding_ledger(answer, {sentences[0]: [1], sentences[1]: [2]})
    assert ledger.status == "complete"
    assert ledger.total_sentences == 2
    assert ledger.unsupported_sentences == 0
    assert ledger.coverage_ratio == 1.0


def test_missing_sentence_citation_is_incomplete():
    answer = "The system processed 1200 records. The report was published in 2025."
    sentences = split_answer_sentences(answer)
    ledger = build_grounding_ledger(answer, {sentences[0]: [1], sentences[1]: []})
    assert ledger.status == "incomplete"
    assert ledger.unsupported_sentences == 1
    assert ledger.coverage_ratio == 0.5


def test_empty_answer_is_incomplete():
    ledger = build_grounding_ledger("", {})
    assert ledger.status == "incomplete"
    assert ledger.total_sentences == 0


def test_invalid_citation_ids_are_ignored():
    answer = "The system processed 1200 records."
    sentence = split_answer_sentences(answer)[0]
    ledger = build_grounding_ledger(answer, {sentence: [0, -1, "2"]})
    assert ledger.status == "incomplete"
    assert ledger.sentences[0].citation_ids == ()


def test_duplicate_ids_are_deduplicated():
    answer = "The system processed 1200 records."
    sentence = split_answer_sentences(answer)[0]
    ledger = build_grounding_ledger(answer, {sentence: [2, 1, 2]})
    assert ledger.status == "complete"
    assert ledger.sentences[0].citation_ids == (1, 2)
