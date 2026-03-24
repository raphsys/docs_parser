# Cartographie de corpus

- PDF: `/home/raphael/Mes_Projets/docs_parser/tests/doc_pdf/test_docintelligence.pdf`
- Pages: `480`

## Familles de pages

- `body_with_figure`: 176 pages. Exemples: 26, 28, 30, 32, 33, 34, 35, 37, 38, 39, 40, 45
- `unknown`: 100 pages. Exemples: 1, 2, 4, 5, 13, 23, 29, 36, 41, 42, 46, 77
- `body_text_two_column`: 58 pages. Exemples: 3, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27
- `body_text_two_column_equations`: 58 pages. Exemples: 6, 62, 69, 70, 92, 105, 106, 116, 117, 142, 149, 161
- `body_text_two_column_sectioned`: 40 pages. Exemples: 25, 31, 44, 48, 72, 83, 90, 114, 140, 174, 179, 185
- `toc`: 34 pages. Exemples: 7, 8, 9, 10, 11, 12, 43, 84, 102, 108, 109, 110
- `table_page`: 4 pages. Exemples: 131, 163, 240, 257
- `body_text_single_column_sparse`: 3 pages. Exemples: 112, 358, 394
- `mixed_page`: 3 pages. Exemples: 82, 126, 355
- `body_text`: 2 pages. Exemples: 186, 435
- `body_with_diagram`: 2 pages. Exemples: 50, 89

## Roles de page

- `body`: 446 pages
- `toc`: 34 pages

## Signatures dominantes

- `body|body_text_two_column|cols=2|two_col,header_footer`: 53 pages. Exemples: 14, 15, 16, 17, 18, 19
- `body|body_text_two_column_sectioned|cols=2|two_col,header_footer,section_heading`: 40 pages. Exemples: 25, 31, 44, 48, 72, 83
- `body|body_with_figure|cols=2|two_col,equation,caption,header_footer`: 32 pages. Exemples: 30, 32, 60, 61, 74, 97
- `body|body_text_two_column_equations|cols=2|two_col,equation,header_footer`: 32 pages. Exemples: 69, 70, 92, 105, 106, 116
- `body|body_text_two_column_equations|cols=2|two_col,equation,header_footer,section_heading`: 23 pages. Exemples: 6, 117, 142, 149, 168, 169
- `body|body_with_figure|cols=2|two_col,caption,header_footer,section_heading`: 23 pages. Exemples: 33, 52, 58, 68, 98, 125
- `body|body_with_figure|cols=2|two_col,equation,caption,header_footer,non_text_dense`: 19 pages. Exemples: 38, 39, 47, 122, 130, 137
- `body|unknown|cols=2|two_col,header_footer`: 15 pages. Exemples: 41, 194, 230, 249, 258, 267
- `body|body_with_figure|cols=2|two_col,caption,header_footer`: 15 pages. Exemples: 51, 87, 138, 173, 178, 271
- `body|body_with_figure|cols=2|two_col,equation,caption,header_footer,section_heading`: 13 pages. Exemples: 59, 76, 79, 80, 86, 104
- `body|body_with_figure|cols=2|two_col,caption,header_footer,non_text_dense`: 12 pages. Exemples: 35, 37, 45, 121, 153, 188
- `body|body_with_figure|cols=2|two_col,caption,header_footer,section_heading,non_text_dense`: 11 pages. Exemples: 26, 28, 49, 67, 146, 382
- `body|body_with_figure|cols=1|equation,caption,header_footer`: 11 pages. Exemples: 55, 63, 65, 73, 133, 225
- `body|unknown|cols=2|two_col,equation,caption,header_footer`: 11 pages. Exemples: 134, 143, 162, 210, 212, 248
- `body|body_with_figure|cols=2|two_col,equation,caption,header_footer,section_heading,non_text_dense`: 10 pages. Exemples: 34, 40, 54, 64, 66, 285
- `toc|toc|cols=2|two_col,header_footer`: 10 pages. Exemples: 467, 468, 469, 470, 471, 472
- `body|unknown|cols=2|two_col,equation,header_footer`: 8 pages. Exemples: 85, 144, 236, 292, 294, 301
- `body|unknown|cols=1|header_footer`: 7 pages. Exemples: 1, 23, 175, 215, 361, 367
- `body|unknown|cols=1|plain`: 5 pages. Exemples: 2, 4, 5, 13, 359
- `body|unknown|cols=1|equation,header_footer,section_heading`: 5 pages. Exemples: 93, 95, 107, 227, 442
- `toc|toc|cols=2|two_col,equation,header_footer`: 5 pages. Exemples: 102, 466, 473, 475, 479
- `body|unknown|cols=2|two_col,equation,caption,header_footer,section_heading`: 5 pages. Exemples: 123, 127, 151, 262, 312
- `body|body_text_two_column|cols=2|two_col`: 4 pages. Exemples: 3, 22, 214, 360
- `body|unknown|cols=2|two_col,header_footer,section_heading`: 4 pages. Exemples: 29, 36, 148, 190
- `body|body_with_figure|cols=2|two_col,equation,caption,diagram_labels,header_footer`: 4 pages. Exemples: 53, 111, 181, 253

## Roles de blocs

- `title`: 2288. Exemples: 2, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18
- `body`: 2068. Exemples: 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14
- `equation_inline`: 758. Exemples: 6, 9, 11, 30, 32, 34, 38, 39, 40, 43, 47, 53
- `header`: 602. Exemples: 7, 8, 9, 10, 11, 12, 15, 18, 19, 23, 25, 26
- `section_heading`: 309. Exemples: 6, 7, 8, 9, 10, 11, 12, 25, 26, 28, 29, 31
- `figure_caption`: 276. Exemples: 26, 28, 30, 32, 33, 34, 35, 37, 38, 39, 40, 42
- `diagram_text_label`: 99. Exemples: 43, 50, 53, 62, 78, 84, 88, 94, 96, 99, 100, 108
- `diagram_label`: 37. Exemples: 50, 53, 62, 84, 88, 89, 96, 99, 100, 115, 120, 128
- `footer`: 25. Exemples: 1, 6, 14, 16, 17, 20, 21, 24, 57, 113, 166, 212
