all: slides.pdf

%.pdf: %.tex %.aux
	pdflatex $<

%.aux: %.tex
	pdflatex $<

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *~ *.synctex.gz

