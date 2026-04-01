# Alzheimer's Disease Progression Modeling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red.svg)](https://streamlit.io)

> Interactive Streamlit demo illustrating how a deep-learning pipeline *could* predict Alzheimer's disease progression from longitudinal brain MRI.

## What This App Demonstrates

- **Synthetic patient simulation** -- adjust age, APOE4 status, baseline MMSE, and hippocampal volume via sidebar sliders.
- **Risk stratification** -- a simple rule-based formula classifies patients as Slow, Moderate, or Rapid progressors.
- **Cognitive trajectory visualization** -- projected MMSE decline curves for each risk category.
- **Brain atrophy rendering** -- procedurally generated 2-D "MRI slices" that expand ventricles and shrink hippocampi over time.
- **Architecture overview** -- diagram of a hypothetical 3D CNN-LSTM pipeline (not trained or executed in the app).

## Running Locally

```bash
pip install streamlit numpy matplotlib scipy
streamlit run app.py
```

## Limitations

- **No real data.** All patient profiles and brain images are synthetic / procedurally generated. No ADNI, OASIS, or other clinical dataset is loaded or used.
- **No trained model.** Risk scores are computed with a hand-coded weighted sum, not a neural network. The architecture tab describes a plausible design but no model weights exist.
- **Hard-coded metrics.** Numbers such as AUC, sensitivity, specificity, and biomarker correlations shown in the "Clinical Context" tab are illustrative placeholders, not computed from any experiment.
- **Not clinically validated.** This project has no IRB approval, no clinical partnerships, and no peer-reviewed publication.

## References

- Alzheimer's Association. *2024 Alzheimer's Disease Facts and Figures.* <https://www.alz.org/alzheimers-dementia/facts-figures>
- Jack, C.R. et al. (2018). NIA-AA Research Framework. *Alzheimer's & Dementia*, 14(4), 535--562. <https://doi.org/10.1016/j.jalz.2018.02.018>
- Marinescu, R.V. et al. (2020). TADPOLE Challenge. *Alzheimer's & Dementia*, 16(S4). <https://doi.org/10.1002/alz.038668>

## License

MIT License -- see [LICENSE](LICENSE) for details.

**Disclaimer:** This software is for educational and portfolio purposes only. It is not intended for clinical diagnosis or treatment decisions.
