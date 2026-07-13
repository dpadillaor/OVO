# Keep every intermediate file out of the source tree.
# Authoritative: latexmk honours this regardless of the flags LaTeX Workshop passes.
$out_dir = 'build';
$pdf_mode = 1;      # pdflatex
$bibtex_use = 2;    # run biber/bibtex as needed, clean .bbl
