from shared.vocabulary import KEYWORD_CODES, KEYWORD_FEATURES, KEYWORDS, detect_codes, keyword_features


def test_vocabulary_codes_are_unique():
    assert len(KEYWORD_CODES) == len(set(KEYWORD_CODES))
    assert len(KEYWORD_FEATURES) == len(KEYWORD_CODES)


def test_every_keyword_has_a_portuguese_label():
    assert all(keyword.label.strip() for keyword in KEYWORDS)


def test_detects_english_terms_and_bigrams():
    assert {"chest", "pain", "chest-pain", "dyspnea"} <= detect_codes("Chest pain, dyspnea")


def test_keeps_slash_terms_as_single_tokens():
    assert {"n-v", "abd-pain"} <= detect_codes("ABD pain n/v")


def test_detects_portuguese_labels_ignoring_accents_and_case():
    assert "chest-pain" in detect_codes("DOR TORÁCICA há 1 hora")
    assert "chills" in detect_codes("Febre alta e calafrios")


def test_empty_text_detects_nothing():
    assert detect_codes("") == set()
    assert detect_codes(None) == set()


def test_keyword_features_merge_codes_and_text_and_drop_unknown_codes():
    features = keyword_features(["fever", "not-a-code"], "dyspnea")

    assert features["keyword_fever"] == 1
    assert features["keyword_dyspnea"] == 1
    assert sum(features.values()) == 2
    assert set(features) == set(KEYWORD_FEATURES)


def test_a_selected_bigram_also_activates_its_words_like_the_training_text_does():
    features = keyword_features(["chest-pain"], None)

    assert {code for code in ("chest", "pain", "chest-pain") if features[f"keyword_{code}"]} == {"chest", "pain", "chest-pain"}
    assert sum(features.values()) == 3


def test_portuguese_label_match_also_activates_the_words():
    features = keyword_features([], "Dor torácica")

    assert features["keyword_chest"] == features["keyword_pain"] == features["keyword_chest-pain"] == 1
