# dubGPUStack

dubGPUStack empile des images astronomiques sur une carte NVIDIA. On lui donne un dossier de brutes ou une séquence Siril (`.seq`). Il détecte les étoiles, aligne les images, applique un drizzle Bayer, puis combine les poses : rejet, normalisation et fondu, selon la mathématique de Siril 1.4. Le résultat est un fichier FITS.

L'interface est en français, dans le thème de dubDSR et dubFocus. C'est un seul exécutable Windows. Le pilote NVIDIA déjà installé suffit : pas de Python, pas de CUDA Toolkit, pas de redistribuable à installer. La fenêtre s'ouvre aussi sans carte ; l'empilement indique alors que le GPU est indisponible.

On le lance sans argument, ou en lui passant le dossier ou le `.seq`. En ligne de commande, `probe` donne le nom de la carte et `stack` lance l'empilement.

dubGPUStack reprend GPU Stacker (Jason Prater, GPL-3) : les mêmes algorithmes, réécrits en Rust dans un seul programme. Licence GNU GPL-3.0 ou ultérieure. Voir `LICENSE` et `THIRD-PARTY-NOTICES.txt`.
