from translation_engines.translation_memory import TranslationMemory


def test_ignores_explicitly_unvalidated_entries():
    memory = TranslationMemory()
    added = memory.add({
        "source": "Hidden layers",
        "target": "Couches cachées",
        "source_lang": "en",
        "target_lang": "fr",
        "validated": False,
    })
    assert added is False
    assert memory.lookup_exact("Hidden layers", "en", "fr") is None
    assert memory.skipped_unvalidated == 1


def test_accepts_entries_without_validated_flag():
    memory = TranslationMemory()
    # Historical export shape, no 'validated' field -> treated as validated.
    memory.add({
        "source_text": "Activation functions",
        "translated_text": "Fonctions d'activation",
        "source_lang": "en",
        "target_lang": "fr",
        "block_role": "header",
    })
    assert memory.lookup_exact("Activation functions", "en", "fr") == "Fonctions d'activation"
    assert memory.lookup_normalized("  activation   functions ", "en", "fr") == "Fonctions d'activation"
