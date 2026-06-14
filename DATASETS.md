# Datasets — Sources, Licensing & Attribution

> **Important.** The evaluation images shipped with this release are **processed
> derivatives** (resized / cropped / tiled to fixed-size patches) of the original
> public benchmarks, prepared **solely to reproduce the evaluation in this repo**.
> They are **not** the original imagery, and we claim **no ownership** over the
> underlying images. All rights to the source imagery and annotations remain with
> the original dataset providers. By using these files you agree to comply with
> the **original license of each dataset** (summarised below). For any use beyond
> reproducing our reported numbers, please obtain the data from the official
> sources and follow their terms.

We thank the authors of DIOR, DOTA and xView for making these benchmarks
available to the community.

---

## Overview

| Dataset | Underlying imagery | License / usage | Official source |
|---|---|---|---|
| **DIOR**  | Google Earth optical imagery | Academic / research use; cite the benchmark paper | Han et al. (NWPU) |
| **DOTA**  | Google Earth + GF-2 / JL-1 satellites + CycloMedia aerial | **Academic use only — commercial use prohibited**; respect Google Earth terms | https://captain-whu.github.io/DOTA/ |
| **xView** | WorldView-3 satellite (0.3 m GSD) | **CC BY-NC-SA 4.0** — non-commercial, attribution, share-alike; requires accepting the terms of use | https://xviewdataset.org/ |

---

## DIOR

A large-scale benchmark for object **D**etect**I**on in **O**ptical **R**emote
sensing images (20 classes, ~23k images), proposed by Junwei Han's group at
Northwestern Polytechnical University. The imagery is sourced from Google Earth;
use is intended for academic research and requires citing the benchmark paper.

```bibtex
@article{li2020dior,
  title   = {Object detection in optical remote sensing images: A survey and a new benchmark},
  author  = {Li, Ke and Wan, Gang and Cheng, Gong and Meng, Liqiu and Han, Junwei},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {159},
  pages   = {296--307},
  year    = {2020}
}
```

## DOTA

A **L**arge-scale **D**ataset for **O**bject de**T**ection in **A**erial images.
Per the official terms: *"All images and their associated annotations in DOTA can
be used for academic purposes only, but any commercial use is prohibited."* The
images are collected from Google Earth, the GF-2 and JL-1 satellites (China Centre
for Resources Satellite Data and Application) and aerial imagery from CycloMedia
B.V.; users of the Google Earth content must additionally comply with the
[Google Earth terms of use](https://www.google.com/permissions/geoguidelines.html).
This release uses the DOTA-v1.5 base/novel splits for zero-shot evaluation.

```bibtex
@inproceedings{xia2018dota,
  title     = {DOTA: A large-scale dataset for object detection in aerial images},
  author    = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}
@article{ding2021dota2,
  title   = {Object detection in aerial images: A large-scale benchmark and challenges},
  author  = {Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael Ying and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2021}
}
```

## xView

Released by the Defense Innovation Unit Experimental (DIUx) and the National
Geospatial-Intelligence Agency (NGA) under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)** license. It contains high-resolution WorldView-3 satellite
imagery (0.3 m ground sample distance) with 60 classes. **Non-commercial use
only**, with **attribution** and **share-alike**; downloading the original data
requires agreeing to the terms of use at <https://xviewdataset.org/terms.html>.
Full license text: <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

```bibtex
@article{lam2018xview,
  title   = {xView: Objects in context in overhead imagery},
  author  = {Lam, Darius and Kuzma, Richard and McGee, Kevin and Dooley, Samuel and Laielli, Michael and Klaric, Matthew and Bulatov, Yaroslav and McCord, Brendan},
  journal = {arXiv preprint arXiv:1802.07856},
  year    = {2018}
}
```

---

## Removal / takedown

These derivative evaluation files are provided in good faith for research
reproducibility only. If you are a rights holder and would like any content
removed, please contact **Guoting Wei (weiguoting@njust.edu.cn)** and we will
take it down promptly.
