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

## Training Datasets (3 releases)

1. **DIETA-Parallel-ENIT v1** — curated EN–IT bitext from OPUS (**~39.7 GB**)
   • **Download:** https://drive.google.com/file/d/1D5oMNs4AnveCkIyx5FRQUJxOzmSm2riv/view?usp=sharing
2. **DIETA-BT-NewsCrawl v1** — back-translations from NewsCrawl/other news sources (**~17 GB**)
   • **Download:** https://drive.google.com/file/d/1Ewn-9BbupFOeMCSyu-u35vUgJvPvFrrL/view?usp=sharing
3. **DIETA-BT-FineWeb v1** — back-translations from FineWeb (**~25.4 GB**)
   • **Download:** https://drive.google.com/file/d/1H-4yRPMFBK64HboaL0uZGrtyxtwNbfqJ/view?usp=sharing

## Evaluation Set

Paired **EN source** and **IT reference** used in the paper's evaluations (450 sentences).

* **English (source) — Download:** https://drive.google.com/file/d/1nFhT1mPs4W8wiz6byVYzCS7GEGToYBCt/view?usp=sharing
* **Italian (human post-edited/reference) — Download:** https://drive.google.com/file/d/1GGjixOnOOc_vyEI2BhV9WFswKggf8yVV/view?usp=sharing


## Prompting format

During training we used **explicit direction tags** (minimal prefix format):

```
ENG: <english sentence> IT: <italian translation>
IT: <italian sentence> ENG: <english translation>
```

Use the same scheme at inference for best results.

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
