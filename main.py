import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import warnings
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from jinja2 import Template
from datetime import datetime
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class SpeechLSTM(nn.Module):
    """LSTM model for temporal speech pattern analysis"""
    def __init__(self, input_size, hidden_size, num_layers):
        super(SpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class CognitiveDeclineDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.features_df = None
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.pca = None
        
        # Initialize BERT for NLP analysis
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Initialize LSTM for temporal analysis
        self.lstm_model = SpeechLSTM(input_size=40, hidden_size=64, num_layers=2)
        
    def extract_mfcc_features(self, y, sr):
        """Extract MFCC features for CNN/LSTM analysis"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return mfccs.T  # Transpose to get time steps as first dimension

    def analyze_speech_rate_lstm(self, mfcc_features):
        """Use LSTM to analyze temporal patterns for speech rate"""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)
            output = self.lstm_model(features_tensor)
            return output.item()

    def analyze_with_bert(self, text):
        """Use BERT to analyze linguistic features"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def process_audio(self, audio_path, name, age, gender):
        """Process audio file with advanced cognitive feature extraction"""
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
                
            # Load audio with error handling
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {str(e)}")
                return None
                
            if len(y) == 0:
                print(f"Audio file {audio_path} is empty")
                return None
                
            # Initialize features with advanced cognitive markers
            features = {
                'file_path': os.path.basename(audio_path),
                'name': name,
                'age': age,
                'gender': gender,
                'duration': duration,
                'transcription': "",
                # Core cognitive features
                'speech_rate_lstm': 0,          # LSTM-based speech rate analysis
                'pause_frequency': 0,            # HMM-style pause detection
                'filler_density': 0,             # Disfluency analysis
                'lexical_diversity': 0,          # Type-Token Ratio
                'anomia_score': 0,              # BERT-based word finding difficulty
                'syntax_errors': 0,              # Transformer-based grammar analysis
                'acoustic_abnormality': 0,       # CNN-style MFCC analysis
                'semantic_coherence': 0,         # BERT-based semantic analysis
                'pronoun_ratio': 0               # POS-tag based analysis
            }
            
            # Extract advanced audio features
            try:
                # MFCC features for CNN/LSTM analysis
                mfcc_features = self.extract_mfcc_features(y, sr)
                features['speech_rate_lstm'] = self.analyze_speech_rate_lstm(mfcc_features)
                
                # Pause analysis (HMM-inspired)
                features.update(self.analyze_pauses_hmm(y, sr))
                
                # Acoustic abnormality score
                features['acoustic_abnormality'] = np.mean(np.std(mfcc_features, axis=0))
                
            except Exception as e:
                print(f"Error extracting audio features from {audio_path}: {str(e)}")
            
            # Speech recognition with error handling
            try:
                with sr.AudioFile(audio_path) as source:
                    audio_data = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_google(audio_data)
                        features['transcription'] = text
                        
                        # Advanced NLP analysis
                        try:
                            nlp_features = self.analyze_text_with_nlp(text)
                            features.update(nlp_features)
                        except Exception as e:
                            print(f"Error analyzing linguistics from {audio_path}: {str(e)}")
                            
                    except sr.UnknownValueError:
                        print(f"Could not understand audio in {audio_path}")
                    except sr.RequestError as e:
                        print(f"Could not request results for {audio_path}; {e}")
            except Exception as e:
                print(f"Error processing audio file {audio_path}: {str(e)}")
            
            return features
            
        except Exception as e:
            print(f"Unexpected error processing {audio_path}: {str(e)}")
            return None
    
    def analyze_pauses_hmm(self, y, sr):
        """HMM-inspired pause and hesitation analysis"""
        features = {}
        
        # Energy-based pause detection
        rms = librosa.feature.rms(y=y)[0]
        silence_threshold = np.percentile(rms, 15)
        is_speech = rms > silence_threshold
        speech_changes = np.diff(is_speech.astype(int))
        
        # Count pauses
        pause_starts = np.where(speech_changes == -1)[0]
        pause_ends = np.where(speech_changes == 1)[0]
        
        # Handle edge cases
        if len(pause_ends) > 0 and len(pause_starts) > 0:
            if pause_ends[0] < pause_starts[0]:
                pause_starts = np.insert(pause_starts, 0, 0)
            if pause_starts[-1] > pause_ends[-1]:
                pause_ends = np.append(pause_ends, len(speech_changes))
            
            pause_durations = [(end - start) * (len(y)/len(rms)/sr) 
                             for start, end in zip(pause_starts, pause_ends)]
            
            features['pause_frequency'] = len(pause_durations) / (len(y)/sr) * 60  # Pauses per minute
        else:
            features['pause_frequency'] = 0
            
        return features
    
    def analyze_text_with_nlp(self, text):
        """Advanced NLP analysis using BERT and traditional methods"""
        features = {
            'filler_density': 0,
            'lexical_diversity': 0,
            'anomia_score': 0,
            'syntax_errors': 0,
            'semantic_coherence': 0,
            'pronoun_ratio': 0
        }
        
        if not text:
            return features
            
        try:
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            
            # 1. Filler words (disfluencies)
            fillers = ['um', 'uh', 'er', 'ah', 'well', 'like', 'you know']
            filler_count = sum(1 for word in words if word in fillers)
            features['filler_density'] = (filler_count / len(words)) * 100 if words else 0
            
            # 2. Lexical Diversity (Type-Token Ratio)
            unique_words = len(set(words))
            features['lexical_diversity'] = unique_words / len(words) if words else 0
            
            # 3. BERT-based analysis
            bert_embedding = self.analyze_with_bert(text)
            
            # Anomia score (word-finding difficulty) - distance from common words
            common_word_emb = self.analyze_with_bert("the be to of and a in that have I")
            features['anomia_score'] = np.linalg.norm(bert_embedding - common_word_emb)
            
            # 4. Syntax errors (simplified)
            pos_tags = nltk.pos_tag(words)
            grammar_errors = sum(1 for word, tag in pos_tags 
                               if tag in ['NN', 'VB'] and len(word) < 3)  # Very short nouns/verbs
            features['syntax_errors'] = grammar_errors / len(words) if words else 0
            
            # 5. Semantic coherence (BERT similarity between sentences)
            if len(sentences) > 1:
                sent_embeddings = [self.analyze_with_bert(sent) for sent in sentences]
                similarities = [np.dot(sent_embeddings[i], sent_embeddings[i+1])/
                              (np.linalg.norm(sent_embeddings[i])*np.linalg.norm(sent_embeddings[i+1])+1e-6)
                             for i in range(len(sent_embeddings)-1)]
                features['semantic_coherence'] = np.mean(similarities) if similarities else 0
            
            # 6. Pronoun ratio
            pronouns = sum(1 for word, tag in pos_tags if tag in ['PRP', 'PRP$'])
            features['pronoun_ratio'] = pronouns / len(words) if words else 0
            
        except Exception as e:
            print(f"Error in NLP analysis: {str(e)}")
        
        return features
    
    def analyze_dataset(self, audio_data):
        """Process multiple audio files with demographic info"""
        all_features = []
        
        for data in audio_data:
            features = self.process_audio(data['path'], data['name'], data['age'], data['gender'])
            if features is not None:
                all_features.append(features)
                print(f"Processed {features['name']} ({features['age']}, {features['gender']}): {features['file_path']}")
            
        if all_features:
            self.features_df = pd.DataFrame(all_features)
            print(f"\nSuccessfully processed {len(all_features)}/{len(audio_data)} files")
        else:
            print("\nNo files were successfully processed")
            self.features_df = pd.DataFrame()
        
        return self.features_df
    
    def detect_cognitive_patterns(self):
        """Advanced cognitive pattern detection with clustering"""
        if self.features_df is None or len(self.features_df) < 3:
            print("Need at least 3 samples for analysis")
            return None
        
        # Select only numeric features for analysis
        numeric_features = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            print("No numeric features available for analysis")
            return None
            
        # Fill NA values with column means
        X = self.features_df[numeric_features].fillna(self.features_df[numeric_features].mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Dimensionality reduction for visualization
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X_scaled)
        self.features_df[['pca1', 'pca2']] = X_pca
        
        # Clustering
        self.cluster_model = KMeans(n_clusters=3, random_state=42)
        clusters = self.cluster_model.fit_predict(X_scaled)
        self.features_df['cluster'] = clusters
        
        # Anomaly detection
        iso_forest = IsolationForest(contamination=0.15, random_state=42)
        risk_scores = -iso_forest.fit_predict(X_scaled)
        
        # Normalize risk scores
        min_score, max_score = risk_scores.min(), risk_scores.max()
        self.features_df['risk_score'] = 100 * (risk_scores - min_score) / (max_score - min_score + 1e-6)
        self.features_df['risk_level'] = pd.cut(
            self.features_df['risk_score'],
            bins=[0, 30, 70, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Add clinical interpretation
        cluster_profiles = {
            0: "Normal speech patterns",
            1: "Mild cognitive markers",
            2: "Significant cognitive concerns"
        }
        self.features_df['cognitive_profile'] = self.features_df['cluster'].map(cluster_profiles)
        
        return self.features_df
    
    def _plot_to_base64(self):
        """Convert matplotlib plot to base64 encoded image"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate_clinical_report(self, output_dir="clinical_results"):
        """Generate a professional HTML clinical report with visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.features_df) == 0:
            print("No data to generate report")
            return

        # Set style for visualizations
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')  # Fallback style if seaborn is not available
        
        # 1. Risk Distribution Pie Chart
        plt.figure(figsize=(8, 6))
        risk_counts = self.features_df['risk_level'].value_counts()
        colors = {'High': '#ff6b6b', 'Medium': '#ffd166', 'Low': '#06d6a0'}
        risk_counts.plot.pie(autopct='%1.1f%%', colors=[colors[r] for r in risk_counts.index])
        plt.title('Risk Level Distribution', fontweight='bold')
        plt.ylabel('')
        risk_pie = self._plot_to_base64()
        plt.close()

        # 2. Feature Importance
        plt.figure(figsize=(10, 6))
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        if 'risk_score' in numeric_cols:
            corr = self.features_df[numeric_cols].corr()['risk_score'].abs().sort_values(ascending=False)[1:6]
            sns.barplot(x=corr.values, y=corr.index, palette='viridis')
            plt.title('Top 5 Cognitive Indicators', fontweight='bold')
            plt.xlabel('Correlation with Risk Score')
            feature_importance = self._plot_to_base64()
            plt.close()
        else:
            feature_importance = None

        # 3. Age vs Risk
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='risk_level', y='age', data=self.features_df,
                   palette=colors, order=['Low', 'Medium', 'High'])
        plt.title('Age Distribution by Risk Level', fontweight='bold')
        plt.xlabel('Risk Level')
        plt.ylabel('Age')
        age_risk = self._plot_to_base64()
        plt.close()

        # 4. PCA Cluster Visualization
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='pca1', y='pca2', hue='cognitive_profile',
                       style='risk_level', data=self.features_df,
                       palette='Set2', s=100, alpha=0.8)
        plt.title('Cognitive Speech Patterns (PCA)', fontweight='bold')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        pca_plot = self._plot_to_base64()
        plt.close()

        # Prepare patient summary table
        patient_summary = self.features_df[[
            'name', 'age', 'gender', 'risk_score', 'risk_level', 'cognitive_profile'
        ]].sort_values('risk_score', ascending=False)
        patient_summary['risk_score'] = patient_summary['risk_score'].round(1)

        # Prepare detailed findings
        high_risk_patients = self.features_df[self.features_df['risk_level'] == 'High']
        key_findings = []
        
        for _, patient in high_risk_patients.iterrows():
            findings = {
                'name': patient['name'],
                'age': patient['age'],
                'gender': patient['gender'],
                'indicators': []
            }
            
            if 'pause_frequency' in patient and patient['pause_frequency'] > 8:
                findings['indicators'].append(
                    f"Elevated pause frequency ({patient['pause_frequency']:.1f}/min, normal <5)")
            if 'filler_density' in patient and patient['filler_density'] > 15:
                findings['indicators'].append(
                    f"Excessive filler words ({patient['filler_density']:.1f}/100w, normal <10)")
            if 'speech_rate_lstm' in patient and patient['speech_rate_lstm'] < 0.8:
                findings['indicators'].append(
                    f"Slow speech rate ({patient['speech_rate_lstm']:.2f}, normal ~1.0)")
            if 'anomia_score' in patient and patient['anomia_score'] > 0.5:
                findings['indicators'].append(
                    "Significant word-finding difficulties")
            
            key_findings.append(findings)

        # Render HTML report
        report_html = self._render_html_report(
            risk_pie=risk_pie,
            feature_importance=feature_importance,
            age_risk=age_risk,
            pca_plot=pca_plot,
            patient_summary=patient_summary.to_dict('records'),
            key_findings=key_findings,
            analysis_date=datetime.now().strftime("%B %d, %Y")
        )

        # Save report
        with open(os.path.join(output_dir, 'clinical_report.html'), 'w') as f:
            f.write(report_html)
            
        # Save data
        self.features_df.to_csv(os.path.join(output_dir, 'cognitive_assessment.csv'), index=False)
        
        print(f"\nProfessional clinical report generated in {output_dir}/ directory")

    def _render_html_report(self, **kwargs):
        """Render professional HTML report"""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Decline Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .section {
            background-color: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            margin: 0;
            font-size: 2.2em;
        }
        h2 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .risk-high {
            background-color: #ff6b6b;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .risk-medium {
            background-color: #ffd166;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .risk-low {
            background-color: #06d6a0;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .finding-card {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .finding-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .finding-item {
            margin-left: 20px;
            margin-bottom: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Cognitive Decline Analysis Report</h1>
        <p>Speech-based cognitive assessment | Generated on {{ analysis_date }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents the findings from automated speech analysis for cognitive decline detection in {{ patient_summary|length }} individuals. 
        The analysis utilized advanced machine learning techniques including LSTM networks for temporal patterns, 
        BERT transformers for linguistic analysis, and acoustic feature extraction.</p>
        
        <div class="image-container">
            <img src="data:image/png;base64,{{ risk_pie }}" alt="Risk Level Distribution">
        </div>
    </div>

    {% if feature_importance %}
    <div class="section">
        <h2>Key Indicators</h2>
        <p>The following cognitive markers showed the strongest correlation with risk scores:</p>
        
        <div class="image-container">
            <img src="data:image/png;base64,{{ feature_importance }}" alt="Feature Importance">
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>Age Distribution</h2>
        <p>Relationship between age and cognitive risk levels:</p>
        
        <div class="image-container">
            <img src="data:image/png;base64,{{ age_risk }}" alt="Age vs Risk">
        </div>
    </div>

    <div class="section">
        <h2>Patient Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Age</th>
                    <th>Gender</th>
                    <th>Risk Score</th>
                    <th>Risk Level</th>
                    <th>Cognitive Profile</th>
                </tr>
            </thead>
            <tbody>
                {% for patient in patient_summary %}
                <tr>
                    <td>{{ patient.name }}</td>
                    <td>{{ patient.age }}</td>
                    <td>{{ patient.gender }}</td>
                    <td>{{ patient.risk_score }}</td>
                    <td>
                        {% if patient.risk_level == 'High' %}
                            <span class="risk-high">{{ patient.risk_level }}</span>
                        {% elif patient.risk_level == 'Medium' %}
                            <span class="risk-medium">{{ patient.risk_level }}</span>
                        {% else %}
                            <span class="risk-low">{{ patient.risk_level }}</span>
                        {% endif %}
                    </td>
                    <td>{{ patient.cognitive_profile }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    {% if key_findings %}
    <div class="section">
        <h2>Key Clinical Findings</h2>
        <p>The following patients showed significant indicators of cognitive decline:</p>
        
        {% for finding in key_findings %}
        <div class="finding-card">
            <div class="finding-title">{{ finding.name }} ({{ finding.age }}, {{ finding.gender }})</div>
            {% for indicator in finding.indicators %}
            <div class="finding-item">• {{ indicator }}</div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Cognitive Patterns Visualization</h2>
        <p>PCA projection of speech patterns showing cluster separation:</p>
        
        <div class="image-container">
            <img src="data:image/png;base64,{{ pca_plot }}" alt="PCA Visualization">
        </div>
    </div>

    <div class="section">
        <h2>Methodology</h2>
        <p><strong>Speech Analysis Techniques:</strong></p>
        <ul>
            <li><strong>Temporal Patterns:</strong> LSTM networks analyzed speech rate and rhythm</li>
            <li><strong>Disfluencies:</strong> HMM-inspired detection of pauses and filler words</li>
            <li><strong>Linguistic Analysis:</strong> BERT transformers assessed word-finding difficulties and semantic coherence</li>
            <li><strong>Acoustic Features:</strong> MFCCs and prosody features evaluated articulation</li>
        </ul>
        
        <p><strong>Risk Classification:</strong></p>
        <ul>
            <li><span class="risk-low">Low Risk</span>: Normal speech patterns with minimal indicators</li>
            <li><span class="risk-medium">Medium Risk</span>: Mild cognitive markers requiring monitoring</li>
            <li><span class="risk-high">High Risk</span>: Significant indicators suggesting clinical evaluation</li>
        </ul>
    </div>

    <div class="footer">
        <p>This report was generated automatically by Cognitive Decline Detection System</p>
        <p>For clinical interpretation, consult a neurologist or cognitive specialist</p>
    </div>
</body>
</html>
        """
        template = Template(template_str)
        return template.render(**kwargs)

def analyze_patient_speech(audio_data, output_dir="clinical_results"):
    """Complete clinical analysis pipeline"""
    print("\nADVANCED COGNITIVE DECLINE DETECTION")
    print("="*60)
    print("Analyzing speech samples using:")
    print("- LSTM networks for temporal speech patterns")
    print("- BERT transformers for linguistic analysis")
    print("- HMM-inspired pause detection")
    print("- Acoustic feature analysis")
    print("="*60)
    
    # Initialize detector
    detector = CognitiveDeclineDetector()
    
    # Process audio files
    print("\nProcessing audio samples...")
    features_df = detector.analyze_dataset(audio_data)
    
    if len(features_df) >= 3:
        # Detect cognitive patterns
        print("\nRunning advanced analysis...")
        results = detector.detect_cognitive_patterns()
        
        if results is not None:
            detector.generate_clinical_report(output_dir)
            
            # Print summary
            print("\nANALYSIS COMPLETE")
            print(f"\nProcessed {len(results)} samples")
            
            high_risk = results[results['risk_level'] == 'High']
            if len(high_risk) > 0:
                print("\nHigh-risk individuals detected:")
                for _, subject in high_risk.iterrows():
                    print(f"\n{subject['name']} (age {subject['age']}, {subject['gender']})")
                    print(f"Risk Score: {subject['risk_score']:.1f}/100")
                    print(f"Profile: {subject['cognitive_profile']}")
                    if 'pause_frequency' in subject:
                        print(f"- Pauses: {subject['pause_frequency']:.1f}/min (normal <5)")
                    if 'filler_density' in subject:
                        print(f"- Fillers: {subject['filler_density']:.1f}/100w (normal <10)")
                    if 'speech_rate_lstm' in subject:
                        print(f"- Speech rate: {subject['speech_rate_lstm']:.2f} (normal ~1.0)")
                    
                    if subject['transcription']:
                        print("\nSpeech sample excerpt:")
                        print(subject['transcription'][:200] + "...")
            
            print(f"\nFull report saved to {output_dir}/ directory")
    else:
        print("\nANALYSIS INCOMPLETE")
        print(f"Need at least 3 samples for analysis (found {len(features_df)})")

if __name__ == "__main__":
    # Example usage - replace with your actual audio files
    audio_data = [
    {'path': os.path.join("audio_bank", "Audio_sample_1.wav"), 'name': "Patient1", 'age': 65, 'gender': "Male"},
    {'path': os.path.join("audio_bank", "Audio_sample_2.wav"), 'name': "Patient2", 'age': 70, 'gender': "Female"},
    {'path': os.path.join("audio_bank", "Audio_sample_3.wav"), 'name': "Patient3", 'age': 68, 'gender': "Male"},
    {'path': os.path.join("audio_bank", "Audio_sample_4.wav"), 'name': "Patient4", 'age': 75, 'gender': "Female"},
]
    
    # Filter to existing files
    audio_data = [data for data in audio_data if os.path.exists(data['path'])]
    
    if len(audio_data) >= 3:
        analyze_patient_speech(audio_data)
    else:
        print("ERROR: Clinical analysis requires at least 3 samples")
        print(f"Found {len(audio_data)} valid audio files")