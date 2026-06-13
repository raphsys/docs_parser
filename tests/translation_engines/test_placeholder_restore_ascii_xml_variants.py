from pagetranslate.protection import audit_placeholders, restore_text


def _protection():
    return {"placeholder": '<nt id="PT0001"/>', "text": "https://example.org/x", "kind": "url", "start": 0, "end": 0}


def test_restore_tolerates_ascii_xml_variants():
    protection = _protection()
    variants = [
        'See <nt id="PT0001"/> here',
        'See < nt id = "PT0001" /> here',
        "See <nt id='PT0001' /> here",
        "See <nt id=PT0001/> here",
    ]
    for text in variants:
        restored = restore_text(text, [protection])
        assert protection["text"] in restored
        assert "PT0001" not in restored
        audit = audit_placeholders(restored, [protection])
        assert audit["placeholder_corruption_count"] == 0


def test_audit_flags_missing_placeholder():
    protection = _protection()
    # Model dropped the placeholder entirely.
    restored = restore_text("See here", [protection])
    audit = audit_placeholders(restored, [protection])
    assert audit["placeholder_corruption_count"] == 1
    assert protection["text"] in audit["missing"]
