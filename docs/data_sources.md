# Data and model sources

- Eye Tracking Dataset for the 12-Lead Electrocardiogram Interpretation of Medical Practitioners and Students, PhysioNet **v1.0.0**, DOI: https://doi.org/10.13026/gsr5-8b11
- PTB-XL, PhysioNet **v1.0.3**, DOI: https://doi.org/10.13026/kfzx-aw45
- Lobachevsky University ECG Database (LUDB): see the cited Kalyakulina et al. 2020 source publication.
- Pretrained 23-output model: Elyamani et al., *Scientific Reports* 2024, DOI: https://doi.org/10.1038/s41598-024-72382-3. The original experiment expected the local model at `12 leads.h5`.

The above sources are not bundled here. The local project folder is expected to contain `ptb-xl/`, `eye_tracking/`, `final_digitisation_v9/`, `definitive_clean_gaze/`, and the permitted pretrained model file. Use the exact version and read the license/terms on each original resource before republishing any part of it.

The publication uses ten image-derived pseudo-waveform ECGs with lead order `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`; these are not original raw voltage waveforms.

The manuscript does not claim external diagnostic validation on PTB-XL fold 10: it is held-out **source-domain** evaluation of a PTB-XL pretrained model.
