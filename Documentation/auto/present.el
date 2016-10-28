(TeX-add-style-hook
 "present"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "graphicx"
    "booktabs")
   (LaTeX-add-bibitems
    "p1"))
 :latex)

