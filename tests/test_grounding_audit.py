from grounding_audit import build_grounding_ledger, split_answer_sentences


def test_all_sentences_have_citations():
    answer = "The report lists six team members. The review was held in March 2025."
    sentences = split_answer_sentences(answer)
    ledger = build_grounding_ledger(
        answer,
        {sentences[0]: [1], sentences[1]: [2, 3]},
    )
    assert ledger.status == "complete"
    assert ledger.total_sentences == 2
    assert ledger.supported_sentences == 2
    assert ledger.unsupported_sentences == 0
    assert ledger.coverage_ratio == 1.0


def test_missing_sentence_citation_is_flagged():
    answer = "The report lists six team members. The budget was ten million dollars."
    sentences = split_answer_sentences(answer)
    ledger = build_grounding_ledger(answer, {sentences[0]: [1]})
    assert ledger.status == "incomplete"
    assert ledger.supported_sentences == 1
    assert ledger.unsupported_sentences == 1
    assert ledger.coverage_ratio == 0.5
    assert ledger.sentences[1].citation_ids == ()


def test_invalid_and_duplicate_citation_ids_are_sanitized():
    answer = "The project started in January 2025."
    sentence = split_answer_sentences(answer)[0]
    ledger = build_grounding_ledger(
        answer,
        {sentence: [2, 2, 0, -1, "3"]},
    )
    assert ledger.sentences[0].citation_ids == (2,)


def test_empty_answer_is_incomplete():
    ledger = build_grounding_ledger("", {})
    assert ledger.status == "incomplete"
    assert ledger.total_sentences == 0
    assert ledger.coverage_ratio == 0.0


def test_to_dict_is_json_serializable():
    import json

    answer = "The report lists six team members."
    sentence = split_answer_sentences(answer)[0]
    ledger = build_grounding_ledger(answer, {sentence: [1]})
    encoded = json.dumps(ledger.to_dict())
    assert '"status": "complete"' in encoded
