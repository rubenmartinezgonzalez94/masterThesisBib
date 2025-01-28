````shell
docker run --rm -v /home/ruben/w/masterThesisBib:/workdir -v /home/ruben/w/masterThesisBib/out:/out texlive/texlive:latest pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/out test.tex
````

````shell
docker run --rm -v /home/ruben/w/masterThesisBib:/workdir -v /home/ruben/w/masterThesisBib/out:/out texlive/texlive:latest pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/out thesis_protocol.tex
````