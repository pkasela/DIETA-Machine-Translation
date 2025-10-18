# DIETA-Machine-Translation

Paper at https://clic2025.unica.it/wp-content/uploads/2025/09/51_main_long.pdf

**DIETA** is a compact, decoder-only Transformer (≈0.5B params) trained specifically for high-quality Italian↔English MT, released with five model checkpoints, three training datasets, and one evaluation set. 

## Overview

DIETA focuses on **bidirectional EN↔IT translation** and is trained on a large curated parallel corpus plus synthetic back-translations; we also introduce a small recent **WikiNews-25** evaluation set (450 sentences) for up-to-date news text. 

Code based on the X-Transformers library by Lucidrains:
https://github.com/lucidrains/x-transformers

## Tokenizer (shared across all models)

All DIETA checkpoints use the **Minerva** tokenizer:

* **Tokenizer:** [https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0)

## Models (5 variants)

**Heads-up:** each model checkpoint is **~1.9 GB**. For best performance, we recommend using **`DIETA_allsynth.pt`**.

All checkpoints share the same 0.5B decoder-only backbone; they differ only in the data mixture / continued training schedule. 

* **DIETA (parallel-only)** — trained on direction-tagged EN↔IT bitext
  Weights: https://drive.google.com/file/d/11a_SeSQu5QmuS2-tw_Yc4m_5TIHIEA-K/view?usp=sharing
* **DIETA+BT** — parallel + NewsCrawl back-translations
  Weights: https://drive.google.com/file/d/1MdRcadubEz-ft_vfOyIsrbPA233ONIA6/view?usp=sharing
* **DIETA+cont** — continues **DIETA** for a 2nd epoch on the same mixture
  Weights: https://drive.google.com/file/d/1CnFxGvXfZSnEixVb-jXNgwXdPZjqK6PI/view?usp=sharing
* **DIETA+nosynth** — continues **DIETA** for a 2nd epoch on **parallel only**
  Weights: https://drive.google.com/file/d/1hNFwZLfRQNlRBCUSOp1Azmer4S0vCFpo/view?usp=sharing
* **DIETA+allsynth** — continues **DIETA+cont** for a 3rd epoch on **parallel + NewsCrawl + FineWeb**
  Weights: https://drive.google.com/file/d/1bxhBKY9JGizs4EmMdqNxpeDmLADXM6pN/view?usp=sharing


## Citation

Please cite the following paper if you use the data or code in this repo.

```
@inproceedings{kaselaetal_clicit2025,
    title = "DIETA: A Decoder-only transformer-based model for Italian–English machine TrAnslation",
    author = "Kasela, Pranav and Braga, Marco and Ghiotto, Alessandro and Pilzer, Andrea and Viviani, Marco and Raganato, Alessandro",
    booktitle = "Proceedings of the 11th Italian Conference on Computational Linguistics (CLiC-it 2025)",
    month = sept,
    year = "2025",
    address = "Cagliari, Italy",
    publisher = "CEUR Workshop Proceedings",
}
```
