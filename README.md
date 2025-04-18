# Cognitive_decline_detector

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-ff8800)](https://librosa.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-5c5c5c)](https://cognitive-decline-detector.onrender.com)

## Results

4. Results will be saved in:
- clinical_results/clinical_report.html
- clinical_results/cognitive_assessment.csv
- Visualization plots (bar charts, heatmaps, feature graphs)


## 📂 Directory Structure

cognitive-decline-detection/
├── assets/
│   └── architecture_diagram.png
├── audio_bank/
│   ├── Audio_sample_1.wav
│   ├── Audio_sample_2.wav
│   ├── Audio_sample_3.wav
│   ├── Audio_sample_4.wav
│   └── test_audio.wav
├── clinical_results/
│   ├── clinical_report.html
│   ├── cognitive_assessment.csv
│   └── report_20250416_223315.csv
├── cognitive_results/
│   ├── cognitive_analysis_report.txt
│   ├── cognitive_assessment.csv
│   ├── feature_correlation.png
│   ├── feature_importance.png
│   ├── pca_visualization.png
│   └── risk_distribution.png
├── Supervised_model/
│   ├── metadata.json
│   ├── model.json
│   └── weights.bin
├── api_main.py          # API with FastAPI
├── main.py              # Main execution pipeline
├── test1.py             # Testing script
├── LICENSE              # MIT License
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies

## 📑 License

This project is licensed under the **MIT License**.

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

Full license text available in the [LICENSE](LICENSE) file.

## 🤝 Contributing

Pull requests are welcome! Feel free to open issues for improvements or questions.

## 🙌 Acknowledgments

- 🎵 [Librosa](https://librosa.org)
- 🤗 [HuggingFace Transformers](https://huggingface.co)
- 📘 [NLTK](https://www.nltk.org)
- 📊 [Scikit-learn](https://scikit-learn.org)
- 🎨 [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org)

## 🚀 Future Roadmap

- Add live microphone input
- Fine-tune BERT on neuro-linguistic datasets
- Expand multilingual support
- Deploy with lightweight UI (e.g., Streamlit or Flask)
