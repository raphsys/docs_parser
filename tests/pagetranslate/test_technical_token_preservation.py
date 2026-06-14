from pagetranslate.technical_protection import is_technical_role, technical_tokens


def test_detects_keras_layers_and_shapes():
    toks = technical_tokens("Conv2D output shape (None, 26, 26, 32) with MaxPooling2D")
    assert "Conv2D" in toks and "MaxPooling2D" in toks
    assert any("None" in t for t in toks)


def test_detects_none_and_sql():
    assert "None" in technical_tokens("value is None here")
    assert "SELECT" in technical_tokens("SELECT * FROM users")


def test_detects_function_and_path():
    toks = technical_tokens("call ST_AsText() then open C:\\data\\file.txt")
    assert any("ST_AsText" in t for t in toks)
    assert any("C:\\data" in t for t in toks)


def test_technical_role_flag():
    assert is_technical_role("table_body_cell")
    assert is_technical_role(None, "code")
    assert not is_technical_role("body_paragraph")
