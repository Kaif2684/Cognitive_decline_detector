from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import io
from main import CognitiveDeclineDetector
from typing import Dict, Any
import os

app = FastAPI(
    title="Cognitive Decline Detection API",
    description="API for detecting cognitive decline through speech analysis",
    version="1.1"  # Updated version
)

# Initialize the detector
detector = CognitiveDeclineDetector()

@app.get("/")
def home():
    """API root endpoint with basic information"""
    return {
        "message": "Welcome to the Cognitive Decline Detection API!",
        "endpoints": {
            "/health": "GET - Service health check",
            "/predict": "POST - Quick MFCC-based prediction",
            "/analyze": "POST - Comprehensive cognitive analysis"
        }
    }

@app.get("/health")
def health_check():
    """Endpoint for service health checks (used by deployment platforms)"""
    return {"status": "ok", "service": "cognitive-decline-detector"}

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """
    Quick prediction using MFCC features.
    
    Returns:
        Basic cognitive decline prediction based on MFCC mean threshold.
    """
    try:
        # Validate audio format
        if audio.content_type not in ["audio/wav", "audio/x-wav"]:
            raise HTTPException(
                status_code=400, 
                detail="Only .wav files are supported"
            )
        
        # Read and process audio
        contents = await audio.read()
        y, sr = librosa.load(io.BytesIO(contents), sr=None)
        
        # Get prediction
        prediction = detector.extract_features_and_predict(y, sr)
        
        return {
            "prediction": prediction,
            "analysis_type": "basic_mfcc",
            "threshold": "MFCC mean >= -10 indicates possible decline",
            "file_type": audio.content_type
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/analyze")
async def full_analysis(
    audio: UploadFile = File(...),
    name: str = "Anonymous",
    age: int = 0,
    gender: str = "Unknown"
):
    """
    Comprehensive cognitive analysis.
    
    Args:
        audio: Input audio file (WAV format)
        name: Patient name
        age: Patient age
        gender: Patient gender
    
    Returns:
        Full cognitive analysis with multiple feature scores
    """
    try:
        # Validate audio format
        if audio.content_type not in ["audio/wav", "audio/x-wav"]:
            raise HTTPException(
                status_code=400, 
                detail="Only .wav files are supported"
            )
        
        # Save uploaded file temporarily
        temp_filename = f"temp_audio_{os.urandom(4).hex()}.wav"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await audio.read())
        
        # Process with full cognitive analysis
        result = detector.process_audio(temp_filename, name, age, gender)
        
        # Clean up
        os.remove(temp_filename)
        
        if not result:
            raise HTTPException(
                status_code=400, 
                detail="Audio processing failed - possibly invalid audio content"
            )
            
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)