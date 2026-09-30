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


def test_only_complete_complaints_are_selectable():
    selectable = {keyword.code for keyword in KEYWORDS if keyword.selectable}

    assert {"chest-pain", "toe-pain", "fever", "headache"} <= selectable
    assert not {"toe", "foot", "pain", "chest", "lower", "altered"} & selectable


def test_selectable_labels_are_unique_so_the_user_never_sees_the_same_option_twice():
    labels = [keyword.label for keyword in KEYWORDS if keyword.selectable]

    assert len(labels) == len(set(labels))


def test_selecting_a_complete_complaint_activates_its_internal_fragments():
    features = keyword_features(["toe-pain"], None)

    assert features["keyword_toe"] == features["keyword_pain"] == features["keyword_toe-pain"] == 1


def test_a_longer_portuguese_label_hides_the_shorter_labels_inside_it():
    assert detect_codes("Dor no dedo do pé") == {"toe-pain"}
    assert detect_codes("Dor no dedo") == {"finger-pain"}


def test_the_label_text_sent_with_the_code_adds_nothing_else():
    features = keyword_features(["toe-pain"], "Dor no dedo do pé")
    active = {code for code, value in features.items() if value}

    assert active == {"keyword_toe-pain", "keyword_toe", "keyword_pain"}


def test_a_typed_note_still_adds_what_it_mentions():
    features = keyword_features(["chest-pain"], "Dor torácica — irradia para o braço")

    assert features["keyword_arm"] == 1
