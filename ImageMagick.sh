convert bianco_nero_con_giallo_e_blu.jpg -background none -fill none -fuzz 30% \
  \( -clone 0 +transparent "#0131f8" \) \
  \( -clone 0 +transparent "#16f011" \) \
  \( -clone 0 +transparent "#f7f402" \) \
  \( -clone 0 +transparent "#f62505" \) \
-delete 0 -flatten result.png
