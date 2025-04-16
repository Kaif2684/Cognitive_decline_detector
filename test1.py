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
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Note: Unable to download NLTK resources: {str(e)}")

class CognitiveDeclineDetector:
    def __init__(self):
        """Initialize the cognitive decline detector"""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.features_df = None
        self.scaler = StandardScaler()
        
    def extract_audio_features(self, y, sr):
        """Extract basic audio features"""
        features = {}
        
        # Basic features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['rms_energy'] = np.sqrt(np.mean(y**2))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc)
            features[f'mfcc_{i}_std'] = np.std(mfcc)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        
        # Pitch analysis (simplified)
        if len(y) > 0:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            features['pitch_mean'] = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            features['pitch_std'] = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        return features
    
    def analyze_pauses(self, y, sr):
        """Analyze pauses in speech"""
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
            
            features['pause_count'] = len(pause_durations)
            features['pause_frequency'] = len(pause_durations) / (len(y)/sr) * 60  # Pauses per minute
            features['mean_pause_duration'] = np.mean(pause_durations) if pause_durations else 0
        else:
            features['pause_count'] = 0
            features['pause_frequency'] = 0
            features['mean_pause_duration'] = 0
            
        return features
    
    def analyze_text(self, text):
        """Analyze transcribed text for cognitive markers"""
        features = {}
        
        if not text:
            return {
                'word_count': 0,
                'filler_density': 0,
                'lexical_diversity': 0,
                'pronoun_ratio': 0,
                'sentence_count': 0,
                'incomplete_sentences': 0
            }
            
        # Basic text features
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        
        # Filler word analysis
        fillers = ['um', 'uh', 'er', 'ah', 'well', 'like', 'you know']
        filler_count = sum(1 for word in words if word in fillers)
        features['filler_density'] = (filler_count / len(words)) * 100 if words else 0
        
        # Lexical diversity
        unique_words = len(set(words))
        features['lexical_diversity'] = unique_words / len(words) if words else 0
        
        # Pronoun analysis (simplified)
        pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 
                   'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 
                   'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                   'they', 'them', 'their', 'theirs', 'themselves']
        pronoun_count = sum(1 for word in words if word in pronouns)
        features['pronoun_ratio'] = pronoun_count / len(words) if words else 0
        
        # Incomplete sentences (simplified)
        incomplete_count = sum(1 for s in sentences if len(s.strip()) < 15 or not s.strip()[-1] in ['.', '!', '?'])
        features['incomplete_sentences'] = incomplete_count / len(sentences) if sentences else 0
        
        return features
    
    def process_audio(self, audio_path, name, age, gender):
        """Process audio file and extract cognitive features"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
                
            # Load audio
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
            except FileNotFoundError:
                print(f"Audio file not found: {audio_path}")
                return None
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {str(e)}")
                return None
                
            if len(y) == 0:
                print(f"Audio file {audio_path} is empty")
                return None
                
            # Initialize features
            features = {
                'file_path': os.path.basename(audio_path),
                'name': name,
                'age': age,
                'gender': gender,
                'duration': duration,
                'transcription': ""
            }
            
            # Extract audio features
            try:
                audio_features = self.extract_audio_features(y, sr)
                features.update(audio_features)
                
                # Pause analysis
                pause_features = self.analyze_pauses(y, sr)
                features.update(pause_features)
                
            except Exception as e:
                print(f"Error extracting audio features from {audio_path}: {str(e)}")
            
            # Speech recognition - FIX HERE
            try:
                # Create a temporary WAV file if needed
                temp_wav = None
                use_filepath = audio_path
                
                # Make sure we're using a string path, not an int
                audio_file_path = str(audio_path)
                
                try:
                    with sr.AudioFile(audio_file_path) as source:
                        audio_data = self.recognizer.record(source)
                        try:
                            text = self.recognizer.recognize_google(audio_data)
                            features['transcription'] = text
                            
                            # Text analysis
                            text_features = self.analyze_text(text)
                            features.update(text_features)
                                
                        except sr.UnknownValueError:
                            print(f"Could not understand audio in {audio_path}")
                        except sr.RequestError as e:
                            print(f"Could not request results for {audio_path}; {e}")
                except (ValueError, TypeError) as e:
                    print(f"Speech recognition error with {audio_path}: {str(e)}")
                    # If we can't process directly, we'll still return the features we have
                    
            except Exception as e:
                print(f"Error processing audio file {audio_path}: {str(e)}")
            
            return features
            
        except Exception as e:
            print(f"Unexpected error processing {audio_path}: {str(e)}")
            return None
    
    def analyze_dataset(self, audio_data):
        """Process multiple audio files"""
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
    
    def detect_anomalies(self):
        """Detect anomalies using unsupervised learning"""
        if self.features_df is None or len(self.features_df) < 2:
            print("Need at least 2 samples for analysis")
            return None
        
        # Select only numeric features
        numeric_features = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            print("No numeric features available for analysis")
            return None
            
        # Fill NA values with column means
        X = self.features_df[numeric_features].fillna(self.features_df[numeric_features].mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA for dimensionality reduction
        n_components = min(len(numeric_features), len(self.features_df) - 1, 5)
        if n_components < 1:
            n_components = 1  # Ensure at least one component
            
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA features to dataframe
        for i in range(X_pca.shape[1]):
            self.features_df[f'pca_{i+1}'] = X_pca[:, i]
        
        # Anomaly detection with Isolation Forest
        iso_forest = IsolationForest(contamination=0.2, random_state=42)
        risk_scores = -iso_forest.fit_predict(X_scaled)
        
        # Normalize risk scores
        min_score, max_score = risk_scores.min(), risk_scores.max()
        self.features_df['risk_score'] = 100 * (risk_scores - min_score) / (max_score - min_score + 1e-6)
        self.features_df['risk_level'] = pd.cut(
            self.features_df['risk_score'],
            bins=[0, 30, 70, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Feature importance
        feature_importance = {}
        for feature in numeric_features:
            try:
                corr = np.abs(np.corrcoef(X[feature], self.features_df['risk_score'])[0, 1])
                if not np.isnan(corr):
                    feature_importance[feature] = corr
            except Exception:
                pass  # Skip features that cause errors
        
        # Sort and store feature importance
        self.feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return self.features_df
    
    def generate_report(self, output_dir="cognitive_results"):
        """Generate report with visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.features_df is None or len(self.features_df) == 0:
            print("No data to generate report")
            return
        
        # Set style for visualizations - FIX HERE
        try:
            # Try different style options based on matplotlib version
            for style_name in ['seaborn-v0_8', 'seaborn', 'ggplot', 'default']:
                try:
                    plt.style.use(style_name)
                    break
                except:
                    continue
        except Exception as e:
            print(f"Warning: Could not set plot style: {str(e)}")
        
        # 1. Create a feature correlation heatmap
        try:
            plt.figure(figsize=(12, 10))
            numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
            
            # Filter columns that exist
            cols_to_use = [col for col in ['pause_frequency', 'mean_pause_duration', 
                                          'filler_density', 'lexical_diversity', 'risk_score'] 
                          if col in numeric_cols]
            
            if cols_to_use:
                correlation_matrix = self.features_df[cols_to_use].corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Feature Correlation Matrix', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate correlation heatmap: {str(e)}")
        
        # 2. Risk level distribution
        try:
            plt.figure(figsize=(8, 6))
            risk_counts = self.features_df['risk_level'].value_counts()
            colors = {'High': '#ff6b6b', 'Medium': '#ffd166', 'Low': '#06d6a0'}
            
            # Filter available colors
            plot_colors = [colors[r] for r in risk_counts.index if r in colors]
            
            if len(plot_colors) == len(risk_counts):
                risk_counts.plot.pie(autopct='%1.1f%%', colors=plot_colors)
            else:
                risk_counts.plot.pie(autopct='%1.1f%%')
                
            plt.title('Risk Level Distribution', fontsize=16)
            plt.ylabel('')
            plt.savefig(os.path.join(output_dir, 'risk_distribution.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate risk distribution plot: {str(e)}")
        
        # 3. Feature importance
        try:
            if hasattr(self, 'feature_importance') and self.feature_importance:
                plt.figure(figsize=(10, 6))
                top_features = dict(list(self.feature_importance.items())[:8])
                
                if top_features:
                    sns.barplot(x=list(top_features.values()), y=list(top_features.keys()), palette='viridis')
                    plt.title('Top Cognitive Indicators', fontsize=16)
                    plt.xlabel('Correlation with Risk Score')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()
        except Exception as e:
            print(f"Warning: Could not generate feature importance plot: {str(e)}")
        
        # 4. PCA visualization
        try:
            if 'pca_1' in self.features_df.columns and 'pca_2' in self.features_df.columns:
                plt.figure(figsize=(10, 8))
                colors = {'High': '#ff6b6b', 'Medium': '#ffd166', 'Low': '#06d6a0'}
                
                sns.scatterplot(
                    x='pca_1', y='pca_2', 
                    hue='risk_level', 
                    size='risk_score',
                    data=self.features_df,
                    palette=colors,
                    sizes=(50, 200),
                    alpha=0.8
                )
                plt.title('PCA Visualization of Cognitive Patterns', fontsize=16)
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                
                # Add subject names to points
                for _, row in self.features_df.iterrows():
                    plt.annotate(
                        row['name'],
                        (row['pca_1'], row['pca_2']),
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
                plt.close()
        except Exception as e:
            print(f"Warning: Could not generate PCA visualization: {str(e)}")
        
        # Generate textual report
        report = self._generate_text_report()
        with open(os.path.join(output_dir, 'cognitive_analysis_report.txt'), 'w') as f:
            f.write(report)
        
        # Save data
        self.features_df.to_csv(os.path.join(output_dir, 'cognitive_assessment.csv'), index=False)
        
        print(f"\nAnalysis report generated in {output_dir}/ directory")
    
    def _generate_text_report(self):
        """Generate a detailed text report"""
        report = "MemoTag Speech Intelligence - Cognitive Analysis Report\n"
        report += "=" * 60 + "\n"
        report += f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}\n\n"
        
        # Summary stats
        report += "SUMMARY\n"
        report += "-" * 60 + "\n"
        report += f"Total subjects analyzed: {len(self.features_df)}\n"
        risk_counts = self.features_df['risk_level'].value_counts()
        for level in ['High', 'Medium', 'Low']:
            if level in risk_counts:
                report += f"{level} risk subjects: {risk_counts[level]} " + \
                         f"({risk_counts[level]/len(self.features_df)*100:.1f}%)\n"
        report += "\n"
        
        # Key cognitive indicators
        report += "KEY COGNITIVE INDICATORS\n"
        report += "-" * 60 + "\n"
        if hasattr(self, 'feature_importance'):
            top_features = dict(list(self.feature_importance.items())[:5])
            for feature, importance in top_features.items():
                report += f"{feature}: correlation = {importance:.3f}\n"
        report += "\n"
        
        # Individual assessments (high risk first)
        report += "INDIVIDUAL ASSESSMENTS\n"
        report += "-" * 60 + "\n"
        
        sorted_df = self.features_df.sort_values('risk_score', ascending=False)
        for _, subject in sorted_df.iterrows():
            report += f"Subject: {subject['name']} (Age: {subject['age']}, Gender: {subject['gender']})\n"
            report += f"Risk Score: {subject['risk_score']:.1f}/100 - {subject['risk_level']} Risk\n"
            
            # Notable features
            report += "Notable indicators:\n"
            
            if 'pause_frequency' in subject and subject['pause_frequency'] > 8:
                report += f"- High pause frequency: {subject['pause_frequency']:.1f} pauses/min (normal <5)\n"
                
            if 'filler_density' in subject and subject['filler_density'] > 10:
                report += f"- Elevated filler words: {subject['filler_density']:.1f}% (normal <5%)\n"
                
            if 'lexical_diversity' in subject and subject['lexical_diversity'] < 0.45:
                report += f"- Low lexical diversity: {subject['lexical_diversity']:.2f} (normal >0.5)\n"
                
            if 'incomplete_sentences' in subject and subject['incomplete_sentences'] > 0.3:
                report += f"- Incomplete sentences: {subject['incomplete_sentences']:.2f} (normal <0.2)\n"
            
            # Transcription excerpt
            if 'transcription' in subject and subject['transcription']:
                report += "Speech sample excerpt:\n"
                excerpt = subject['transcription'][:150] + "..." if len(subject['transcription']) > 150 else subject['transcription']
                report += f"\"{excerpt}\"\n"
            
            report += "-" * 40 + "\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS\n"
        report += "-" * 60 + "\n"
        report += "1. Clinical follow-up is recommended for subjects with High risk scores\n"
        report += "2. Consider additional speech tasks for Medium risk subjects\n"
        report += "3. Periodical reassessment (3-6 months) for all subjects to track changes\n\n"
        
        # Limitations
        report += "LIMITATIONS\n"
        report += "-" * 60 + "\n"
        report += "This analysis is a screening tool and not a clinical diagnosis.\n"
        report += "Results should be interpreted by healthcare professionals in context of other assessments.\n"
        report += "Limited sample size may affect the reliability of the risk stratification.\n"
        
        return report

def analyze_patient_speech(audio_data, output_dir="cognitive_results"):
    """Complete cognitive analysis pipeline"""
    print("\nMEMOTAG SPEECH INTELLIGENCE MODULE")
    print("="*60)
    print("Analyzing speech samples for cognitive decline indicators")
    print("="*60)
    
    # Initialize detector
    detector = CognitiveDeclineDetector()
    
    # Process audio files
    print("\nProcessing audio samples...")
    features_df = detector.analyze_dataset(audio_data)
    
    if len(features_df) >= 2:
        # Detect anomalies
        print("\nRunning cognitive pattern detection...")
        results = detector.detect_anomalies()
        
        if results is not None:
            # Generate report
            print("\nGenerating comprehensive report...")
            detector.generate_report(output_dir)
            
            # Print summary
            print("\nANALYSIS COMPLETE")
            print(f"\nProcessed {len(results)} samples")
            
            high_risk = results[results['risk_level'] == 'High']
            if len(high_risk) > 0:
                print("\nHigh-risk individuals detected:")
                for _, subject in high_risk.iterrows():
                    print(f"- {subject['name']} (age {subject['age']}): Risk Score {subject['risk_score']:.1f}/100")
            
            print(f"\nFull report saved to {output_dir}/ directory")
    else:
        print("\nANALYSIS INCOMPLETE")
        print(f"Need at least 2 samples for analysis (found {len(features_df)})")

if __name__ == "__main__":
    # Sample audio data - replace with actual file paths
    audio_data = [
    {'path': os.path.join("audio_bank", "Audio_sample_1.wav"), 'name': "Patient1", 'age': 65, 'gender': "Male"},
    {'path': os.path.join("audio_bank", "Audio_sample_2.wav"), 'name': "Patient2", 'age': 70, 'gender': "Female"},
    {'path': os.path.join("audio_bank", "Audio_sample_3.wav"), 'name': "Patient3", 'age': 68, 'gender': "Male"},
    {'path': os.path.join("audio_bank", "Audio_sample_4.wav"), 'name': "Patient4", 'age': 75, 'gender': "Female"},
]
    
    # Make sure the output directory exists
    os.makedirs("cognitive_results", exist_ok=True)
    
    # Run analysis
    analyze_patient_speech(audio_data)