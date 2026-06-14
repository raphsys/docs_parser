from PIL import Image, ImageDraw

from pagereconstruct.source_text_leak_detector import detect


def _page_with_text():
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([10, 10, 90, 40], fill=(0, 0, 0))  # "text"
    return img


def test_unpatched_area_leaks():
    src = _page_with_text()
    recon = _page_with_text()  # identical -> old text still there
    res = detect(src, recon, [[10, 10, 90, 40]])
    assert res["leak_count"] == 1


def test_patched_area_no_leak():
    src = _page_with_text()
    recon = Image.new("RGB", (200, 100), (255, 255, 255))  # cleaned
    res = detect(src, recon, [[10, 10, 90, 40]])
    assert res["leak_count"] == 0
