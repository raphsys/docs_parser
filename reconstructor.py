import fitz
import os

class DocumentReconstructor:
    """
    Reconstruit un document PDF à partir d'une structure JSON riche.
    Gère la conversion Pixels (200 DPI) vers Points (72 DPI).
    """
    
    def __init__(self):
        # Facteur de conversion : 72 / 200 = 0.36
        self.pixel_to_point = 72.0 / 200.0

    def reconstruct(self, structure, output_path):
        doc = fitz.open()
        
        # Le JSON peut être une liste de pages ou un dict contenant une liste 'pages'
        pages_list = structure.get("pages", []) if isinstance(structure, dict) else structure
        
        for page_data in pages_list:
            # Récupération des dimensions (en pixels dans le JSON)
            px_width = page_data.get("dimensions", {}).get("width", 885)
            px_height = page_data.get("dimensions", {}).get("height", 1110)
            
            # Conversion en points
            pt_width = px_width * self.pixel_to_point
            pt_height = px_height * self.pixel_to_point
            
            page = doc.new_page(width=pt_width, height=pt_height)
            
            # 1. Texte (Couche prioritaire)
            for block in page_data.get("blocks", []):
                if block["type"] == "text":
                    self._insert_text_block(page, block)
                elif block["type"] == "drawing":
                    self._draw_vector(page, block)

        doc.save(output_path)
        doc.close()
        return output_path

    def _insert_text_block(self, page, block):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                style = span.get("style", {})
                text = span.get("text", "")
                bbox = span.get("bbox")
                
                # Conversion couleur Hex -> RGB (0-1)
                color_hex = style.get("color", "#000000").lstrip('#')
                rgb = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                
                # Positionnement : conversion Pixels -> Points
                origin = span.get("origin")
                if origin:
                    pt_origin = fitz.Point(origin[0] * self.pixel_to_point, origin[1] * self.pixel_to_point)
                else:
                    pt_origin = fitz.Point(bbox[0] * self.pixel_to_point, bbox[3] * self.pixel_to_point)
                
                # Taille de police
                fs = style.get("size", 10) * self.pixel_to_point
                
                # Sélection de police
                fontname = self._get_font_name(style)
                
                try:
                    page.insert_text(
                        pt_origin,
                        text,
                        fontsize=fs,
                        fontname=fontname,
                        color=rgb
                    )
                except:
                    # Fallback sur helvetica si la police échoue
                    page.insert_text(pt_origin, text, fontsize=fs, fontname="helv", color=rgb)

    def _get_font_name(self, style):
        flags = style.get("flags", {})
        is_bold = flags.get("bold", False)
        is_italic = flags.get("italic", False)
        is_serif = flags.get("serif", False)
        
        base = "times" if is_serif else "helv"
        if is_bold and is_italic: base += "-bolditalic"
        elif is_bold: base += "-bold"
        elif is_italic: base += "-italic"
        else: base += "-roman" if is_serif else ""
        
        return base

    def _draw_vector(self, page, block):
        # Conversion des coordonnées du rectangle
        bbox = [c * self.pixel_to_point for c in block["bbox"]]
        shape = page.new_shape()
        shape.draw_rect(bbox)
        
        fill = block.get("fill_color")
        stroke = block.get("stroke_color")
        
        if fill and isinstance(fill, (list, tuple)):
            shape.finish(fill=fill, color=stroke, width=0.5)
        else:
            shape.finish(color=(0.8, 0.8, 0.8), width=0.5)
        shape.commit()