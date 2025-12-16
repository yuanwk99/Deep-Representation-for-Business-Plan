# Multimodal Business Plan Analysis for Investment Decision-Making

This repository contains the code for the paper:\
**"Insights into the impact of visual and textual information on investment decision-making: A multimodal business plan analysis via deep representation learning"**

We propose a computational framework to quantify the visual, textual quality of business plans (BPs) using deep pre-trained models, and demonstrate their significant influence on real-world investor decisions.


## 🏗️ Repository Structure

```
├── data_preprocess.py                 # Step 1: Clean raw data & extract text/images
├── text_representation.py             # Step 2: Extract text embeddings (e.g., FinBERT)
├── visual_representation.py           # Step 2: Extract visual embeddings (e.g., DiT/ViT)
├── computing_quality_indicators.py    # Step 3: Compute V_BP, T_BP, I_BP using seed-based similarity
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```


## 📝 Citation

If you use this work in your research, please cite our paper:

```bibtex
Yuan, Weikang, et al. "Insights into the impact of visual and textual information on investment decision-making: A multimodal business plan analysis via deep representation learning." Expert Systems with Applications 296 (2026): 128911.
```


## 📄 License

This work is licensed under a Creative Commons Attribution- NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).
