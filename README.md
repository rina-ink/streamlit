# Instrument Timbre – Interactive Audio Demo

This app previews the **CCMUSIC Database – Instrument Timbre** dataset. 
It allows you to explore instrument timbre ratings, spectrograms, and audio playback directly in the browser. The app defaults to license-safe mode to ensure playback stays true to the original samples.
Audio samples © the original authors. 
Non-commercial use only; no derivatives.

##  Features
- Browse Chinese and Western instrument samples
- Play audio directly in the browser (in-memory only, no downloads)
- Visualize timbre ratings with radar charts (1–9 scale)
- Inspect mel spectrograms per instrument and (optionally) raw waveforms
- Compare two instruments side-by-side with overlay radar and Δ-bar plots
- License-safe mode ON by default (original samples only) 

- **Contrast Check (A vs B)**  
  - Randomly select or manually pick instruments
  - Listen & compare two samples 
  - Overlay radar plots
  - Δ bar chart showing trait differences 
  - Optional waveform + smoothed envelope view 
 - Each card shows:
    - Audio playback (`.wav`, in-memory only)  
    - Mel spectrogram preview  
    - Radar chart of timbre rating*s


## About my Project

This repository contains the implementation of the interactive web app for exploring the instrument timbre characteristics across cultures. It marks the initial release and was developed as part of a broader project that also involved: performing EDA-Analysis, implementing and evaluating multiple ML algorithms. The app enables interactive comparison using both audio perception and visual analytics. 

![project_img](images/instrument_timbre_analysis.png)

🌀 Explore the color of sound