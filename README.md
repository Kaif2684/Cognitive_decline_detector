
# Cognitive_decline_detector

## 🌐 Live API Deployment

You can test the API live at:

https://cognitive-decline-detector.onrender.com

This includes:
- `/health` → Service health check
- `/predict` → Quick MFCC-based prediction
- `/analyze` → Comprehensive cognitive analysis from audio

Deployed on Render (Free Tier) using FastAPI.  
Note: Free instances may take ~30–50 seconds to wake up.

## 📊 Output Location

Results will be saved in:
- `clinical_results/clinical_report.html`
- `clinical_results/cognitive_assessment.csv`
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
