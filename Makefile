
open: main.pdf.open

%.pdf.open: %.pdf
	open $<

%.pdf: %.tex references.bib
	pdflatex $*.tex
	bibtex $*
	pdflatex $*.tex
	pdflatex $*.tex

.SECONDARY:
