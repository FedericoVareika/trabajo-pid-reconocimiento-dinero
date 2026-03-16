# Default directory if none is provided
DIR ?= seguimiento1

# Derived paths based on the DIR variable
TEXFILES := $(wildcard $(DIR)/*.tex)
BIBFILES := $(wildcard $(DIR)/*.bib)
TEXSRC   := $(DIR)/Documentacion.tex
PDFOUT   := $(DIR)/Documentacion.pdf

.PHONY: all clean

# The main target
all: $(PDFOUT)

$(PDFOUT): $(TEXFILES) $(BIBFILES)
	latexmk -pdf -shell-escape -output-directory=$(DIR) $(TEXSRC)

clean:
	latexmk -C -output-directory=$(DIR) $(TEXSRC)
