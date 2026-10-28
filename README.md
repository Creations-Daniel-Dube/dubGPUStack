# dubGPUStack

Empilement d'images astronomiques sur GPU NVIDIA. dubGPUStack reprend [GPU Stacker](python) (Jason Prater, GPL-3) : la mathématique de Siril 1.4 (rejet, normalisation, fondu) et le chemin des brutes (étoiles, registration, drizzle Bayer), dans un seul exécutable Windows.

L'interface est en français. Le thème est celui de dubDSR et dubFocus.

## Ce qu'il faut pour l'utiliser

Le pilote NVIDIA déjà installé suffit. L'exécutable charge `nvcuda.dll`. Il n'y a pas de Python, pas de CUDA Toolkit et pas de redistribuable Visual C++ à installer.

```text
target\release\dubGPUStack.exe
```

La fenêtre s'ouvre même sans carte NVIDIA. L'empilement indique alors clairement que le GPU est indisponible.

## Lancer

- Sans argument : l'interface.
- Un dossier ou un fichier `.seq` : l'interface s'ouvre sur ces données.
- `dubGPUStack probe` : nom de la carte et mémoire, en une ligne JSON. Code 0, ou 1 si le GPU manque.
- `dubGPUStack stack DOSSIER_OU_SEQ` : empilement en ligne de commande.

Options de `stack`, les mêmes que `python -m gpustacker.engine stack` :

```text
--out FICHIER
--rejection winsorized|sigma|none
--sigma BAS HAUT
--weight none|noise|wfwhm|nbstars
--feather PIXELS
--no-norm --no-rgb-equal --no-output-norm
--rows Y0 Y1
--crop X Y W H
--overwrite
--gpu-mem-gib N
--scale 1|1.5|2|3
--pixfrac 0.05..1
--transform affine|similarity
--analyze-only
```

Codes de sortie : 0 réussi, 1 erreur interne, 2 erreur de saisie, 3 annulé.

Les réglages de l'interface sont dans `%APPDATA%\dubGPUStack\settings.json`. Les caches restent à côté des données, dans `gpustacker/` (`analysis.json`, `registration.json`, `measures.json`, `selection.json`, masques `.msk`), au même format que la version Python.

## Compiler

Rust stable pour le programme, et Rust nightly avec la cible `nvptx64-nvidia-cuda` pour les noyaux. Aucun compilateur C ou C++ n'est nécessaire.

```text
rustup toolchain install stable nightly
rustup target add nvptx64-nvidia-cuda --toolchain nightly
cargo build --release
```

Le produit est `target\release\dubGPUStack.exe`. Les noyaux PTX sont compilés par `build.rs` et embarqués dans l'exécutable. Le runtime C est lié en statique (`.cargo/config.toml`).

## Licence

GNU GPL-3.0 ou ultérieure. dubGPUStack est une œuvre dérivée de GPU Stacker et des algorithmes de Siril. Voir `LICENSE` et `THIRD-PARTY-NOTICES.txt`.
