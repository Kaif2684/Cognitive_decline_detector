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
├── audio_bank/                # Input audio files
├── clinical_results/          # Output reports and charts
├── models/                    # Optional model checkpoints
├── main.py                    # Main execution pipeline
├── report_template.html       # Jinja2 HTML report template
├── README.md                  # This file
├── LICENSE                    # MIT License
└── requirements.txt           # Python dependencies


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
