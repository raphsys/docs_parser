from structure_extractor import DocumentParser

# Mock OCR results based on the user's provided dump
mock_results = [
    {"label": "Deep Learning", "bbox": [93, 375, 617, 443], "score": 0.99},
    {"label": "for", "bbox": [274, 466, 316, 495], "score": 0.99},
    {"label": "Vision Systems", "bbox": [92, 508, 642, 575], "score": 0.99},
    {"label": "Mohamed Elgendy", "bbox": [91, 744, 341, 782], "score": 0.99}
]

# Mock Image (needed for visual extractor, but we can pass a dummy or None if we patch it)
from PIL import Image
mock_img = Image.new('RGB', (1107, 1388), color='white')

parser = DocumentParser()

# We need to bypass VisualAttributeExtractor.analyze since it needs a real image for contours
# We'll monkeypatch it for this test
def mock_analyze(self, pil_image, bbox, text=""):
    return [{"texte": text, "style": {"font": "Arial", "size": bbox[3]-bbox[1], "color": "#000000", "flags": {}}, "bbox": bbox, "peak_y": (bbox[1] + bbox[3]) / 2}]

parser.visual_extractor.analyze = mock_analyze.__get__(parser.visual_extractor, type(parser.visual_extractor))

print("--- Running Parse ---")
structure = parser.parse(mock_results, mock_img)

for block in structure:
    print(f"\nBLOCK {block['id']} | bbox={block['bbox']} | spacing={block['line_spacing']:.1f} | p2p={block.get('peak_to_peak_spacing', 0):.1f}")
    for line in block['lines']:
        bbox = line['bbox']
        txt = " ".join([p['texte'] for p in line['phrases']])
        print(f"  Line: {txt} {bbox} | peak_y={line.get('peak_y')}")
