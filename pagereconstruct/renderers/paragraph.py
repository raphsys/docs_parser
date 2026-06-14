from .base import BaseRenderer


class ParagraphRenderer(BaseRenderer):
    renderer_name = "paragraph"


class ListItemRenderer(ParagraphRenderer):
    renderer_name = "list_item"
