rm diplom.aux diplom.aux.blg diplom.bbl diplom.bcf diplom.blg
rm diplom.log diplom.out diplom.run.xml diplom.toc diplom.dvi
pdflatex -shell-escape diplom.tex
pdflatex -shell-escape diplom.tex
biber diplom.aux
pdflatex -shell-escape diplom.tex
pdflatex -shell-escape diplom.tex
