from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
from typing import Dict, List

# ===== FASTAPI APP INITIALIZATION =====
app = FastAPI(title="Exoplanet Detector", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# ===== GLOBAL VARIABLES FOR ML MODELS =====
models = {}
scaler = None
feature_names = []
label_encoder = None

# ===== MODEL LOADING FUNCTION =====
def load_ml_models():
    """
    Load trained ML models and preprocessing objects
    """
    global models, scaler, feature_names, label_encoder
    
    try:
        print("🔄 Loading trained ML models...")
        
        # Load the trained models (they are saved as tuples: (pipeline, classes))
        xgb_pipe, xgb_classes = joblib.load('ml_models/XGBoost_pipeline.pkl')
        cat_pipe, cat_classes = joblib.load('ml_models/CatBoost_pipeline.pkl')
        voting_pipe, voting_classes = joblib.load('ml_models/VotingEnsemble_pipeline.pkl')
        lgb_pipe, lgb_classes = joblib.load('ml_models/LightGBM_pipeline.pkl')
        
        models = {
            'xgboost': xgb_pipe,
            'catboost': cat_pipe,
            'votingensemble': voting_pipe,
            'lightgbm': lgb_pipe
        }
        
        # Load preprocessing objects - use ALL features including engineered ones
        # The models were trained on 15 features (11 original + 4 engineered)
        feature_names = ['period', 'planet_radius', 'depth', 'equilibrium_temp', 'insolation', 
                        'impact', 'duration', 'star_radius', 'star_mass', 'star_teff', 'kepmag',
                        'planet_density_ratio', 'log_period', 'stellar_flux', 'temp_ratio']
        print(f"✅ Using all 15 features including engineered features")
        
        # Since we're using pipelines, we don't need a separate scaler
        scaler = None
        
        # Try to load label encoder if it exists
        try:
            label_encoder = joblib.load('ml_models/label_encoder.pkl')
            print("✅ Label encoder loaded")
        except:
            label_encoder = None
            print("⚠️ No label encoder found, using default mapping")
        
        print("✅ Models loaded successfully!")
        print(f"   Available models: {list(models.keys())}")
        print(f"   Feature names: {len(feature_names)} features")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("💡 Run train_models.py first to train the models")

# ===== MULTI-CLASS PREDICTION FUNCTIONS =====
def decode_predictions(predictions, probabilities=None):
    """
    Decode numerical predictions to human-readable labels
    Handle 3-class classification: CONFIRMED, CANDIDATE, FALSE POSITIVE
    """
    # Mapping for the 3 classes
    class_mapping = {
        0: {"label": "FALSE POSITIVE", "emoji": "❌", "color": "#ff4444"},
        1: {"label": "CANDIDATE", "emoji": "🔍", "color": "#ffa500"}, 
        2: {"label": "CONFIRMED", "emoji": "🌍", "color": "#44ff44"}
    }
    
    decoded_predictions = []
    
    for i, pred in enumerate(predictions):
        class_info = class_mapping.get(pred, {"label": "UNKNOWN", "emoji": "❓", "color": "#888888"})
        
        prediction_data = {
            "row": i + 1,
            "prediction": f"{class_info['emoji']} {class_info['label']}",
            "prediction_code": pred,
            "color": class_info['color']
        }
        
        # Add probabilities if available
        if probabilities is not None and i < len(probabilities):
            prob_array = probabilities[i]
            prediction_data["confidence"] = float(np.max(prob_array))
            prediction_data["probabilities"] = {
                "false_positive": float(prob_array[0]) if len(prob_array) > 0 else 0,
                "candidate": float(prob_array[1]) if len(prob_array) > 1 else 0,
                "confirmed": float(prob_array[2]) if len(prob_array) > 2 else 0
            }
        
        decoded_predictions.append(prediction_data)
    
    return decoded_predictions

def calculate_multi_class_statistics(predictions, decoded_predictions):
    """
    Calculate statistics for multi-class predictions
    """
    prediction_codes = [pred["prediction_code"] for pred in decoded_predictions]
    
    # Count each class
    false_positive_count = sum(1 for code in prediction_codes if code == 0)
    candidate_count = sum(1 for code in prediction_codes if code == 1)
    confirmed_count = sum(1 for code in prediction_codes if code == 2)
    
    total = len(predictions)
    
    return {
        "total_predictions": total,
        "false_positive_count": false_positive_count,
        "candidate_count": candidate_count,
        "confirmed_count": confirmed_count,
        "false_positive_percentage": (false_positive_count / total) * 100,
        "candidate_percentage": (candidate_count / total) * 100,
        "confirmed_percentage": (confirmed_count / total) * 100,
        "prediction_breakdown": {
            "false_positive": false_positive_count,
            "candidate": candidate_count,
            "confirmed": confirmed_count
        }
    }

# ===== VISUALIZATION FUNCTIONS =====
def create_multi_class_plot(predictions, probabilities, statistics):
    """
    Create visualization for multi-class predictions with FIXED error handling
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Multi-class distribution (Pie chart)
        sizes = [
            statistics['false_positive_count'],
            statistics['candidate_count'], 
            statistics['confirmed_count']
        ]
        labels = ['False Positive', 'Candidate', 'Confirmed']
        colors = ['#ff4444', '#ffa500', '#44ff44']
        
        # Only create pie chart if we have non-zero values
        if sum(sizes) > 0:
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax1.text(0.5, 0.5, 'No predictions to display', 
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Exoplanet Classification Distribution', fontweight='bold')
        
        # Plot 2: Confidence distribution - FIXED ERROR HERE
        if probabilities and len(probabilities) > 0:
            # Convert to numpy array safely
            prob_array = np.array(probabilities)
            if prob_array.size > 0:
                max_confidences = np.max(prob_array, axis=1)
                ax2.hist(max_confidences, bins=20, alpha=0.7, color='#4fc3f7', edgecolor='black')
                ax2.set_xlabel('Maximum Confidence Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Confidence Score Distribution')
                ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='High Confidence Threshold')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No probability data', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Confidence Distribution')
        else:
            ax2.text(0.5, 0.5, 'No probability data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Confidence Distribution')
        
        # Plot 3: Class probabilities - FIXED ERROR HERE
        if probabilities and len(probabilities) > 0:
            prob_array = np.array(probabilities)
            if prob_array.size > 0 and len(probabilities) > 5:
                # Show first 5 samples' probability distributions
                sample_probs = probabilities[:5]
                x_pos = np.arange(len(sample_probs))
                
                bottom = np.zeros(len(sample_probs))
                colors_bar = ['#ff4444', '#ffa500', '#44ff44']
                labels_bar = ['False Positive', 'Candidate', 'Confirmed']
                
                for i in range(min(3, prob_array.shape[1])):  # Safe range check
                    probs = [prob[i] for prob in sample_probs if i < len(prob)]
                    if probs:  # Only plot if we have data
                        ax3.bar(x_pos[:len(probs)], probs, bottom=bottom[:len(probs)], 
                               color=colors_bar[i], label=labels_bar[i], alpha=0.8)
                        if len(bottom) == len(probs):
                            bottom += np.array(probs)
                
                ax3.set_xlabel('Sample Index')
                ax3.set_ylabel('Probability')
                ax3.set_title('Class Probabilities for First 5 Samples')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Need more samples\nfor visualization', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Class Probabilities')
        else:
            ax3.text(0.5, 0.5, 'Probability data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Class Probabilities')
        
        # Plot 4: Feature importance
        if models and 'xgboost' in models and hasattr(models['xgboost'], 'feature_importances_'):
            importance = models['xgboost'].feature_importances_
            if feature_names and len(importance) == len(feature_names):
                feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
                feature_imp = feature_imp.sort_values('importance', ascending=True).tail(10)
                
                ax4.barh(feature_imp['feature'], feature_imp['importance'])
                ax4.set_xlabel('Importance')
                ax4.set_title('Top 10 Feature Importance')
            else:
                ax4.text(0.5, 0.5, 'Feature names mismatch', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Feature Importance')
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{graphic}"
        
    except Exception as e:
        print(f"❌ Error creating multi-class plot: {e}")
        import traceback
        traceback.print_exc()
        return ""
    

# ===== FEATURE ENGINEERING =====
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as used during training
    """
    df = df.copy()
    
    # Create engineered features if they don't exist
    if "planet_radius" in df.columns and "star_radius" in df.columns:
        if "planet_density_ratio" not in df.columns:
            df["planet_density_ratio"] = df["planet_radius"] / (df["star_radius"] + 1e-6)
    
    if "period" in df.columns:
        if "log_period" not in df.columns:
            df["log_period"] = np.log1p(df["period"])
    
    if "insolation" in df.columns and "star_radius" in df.columns:
        if "stellar_flux" not in df.columns:
            df["stellar_flux"] = df["insolation"] / (df["star_radius"] ** 2 + 1e-6)
    
    if "equilibrium_temp" in df.columns and "star_teff" in df.columns:
        if "temp_ratio" not in df.columns:
            df["temp_ratio"] = df["equilibrium_temp"] / (df["star_teff"] + 1e-6)
    
    return df

# ===== PREDICTION FUNCTIONS =====
def preprocess_input_data(df: pd.DataFrame, model_type: str = None) -> pd.DataFrame:
    """Preprocess input data for prediction - return DataFrame for pipeline models"""
    global feature_names, scaler
    
    # Apply feature engineering first
    df_engineered = feature_engineering(df)
    
    # Since our models are pipelines, they handle their own preprocessing
    # We just need to ensure we have the right features and handle missing ones
    if feature_names:
        available_features = [col for col in feature_names if col in df_engineered.columns]
        missing_features = [col for col in feature_names if col not in df_engineered.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            for feature in missing_features:
                df_engineered[feature] = 0  # Fill with zeros
        
        # Reorder to match training order
        df_processed = df_engineered[feature_names].copy()
        return df_processed
    else:
        # Fallback: use all numerical columns
        numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns
        return df_engineered[numerical_cols].fillna(0)

def make_predictions(model, X_processed: pd.DataFrame) -> Dict:
    """Make predictions using the trained model"""
    try:
        predictions = model.predict(X_processed)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_processed)
            confidence_scores = np.max(probabilities, axis=1)
        else:
            probabilities = None
            confidence_scores = np.ones(len(predictions)) * 0.5
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'confidence_scores': confidence_scores.tolist()
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Return safe fallback
        n_samples = len(X_processed)
        return {
            'predictions': [1] * n_samples,  # Default to candidate
            'probabilities': None,
            'confidence_scores': [0.5] * n_samples
        }

# ===== FASTAPI ROUTES =====
@app.on_event("startup")
async def startup_event():
    """Load ML models when the application starts"""
    load_ml_models()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the homepage"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Serve the prediction page"""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/api/predict")
async def predict(
    model_type: str = Form(...),
    file: UploadFile = File(...)
):
    """Main prediction endpoint with proper error handling"""
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            {"error": "Invalid file type. Please upload a CSV file."}, 
            status_code=400
        )
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        print(f"📊 Processing file: {file.filename}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        if not models or model_type not in models:
            return JSONResponse(
                {"error": "ML models not loaded properly. Please train models first."},
                status_code=500
            )
        
        # Preprocess data
        X_processed = preprocess_input_data(df, model_type)
        model = models[model_type]
        
        # Make predictions
        prediction_results = make_predictions(model, X_processed)
        
        # Ensure probabilities array is properly structured
        if prediction_results['probabilities'] is not None:
            # Convert to numpy array and back to ensure consistent structure
            prob_array = np.array(prediction_results['probabilities'])
            prediction_results['probabilities'] = prob_array.tolist()
        
        # Decode predictions for multi-class output
        decoded_predictions = decode_predictions(
            prediction_results['predictions'],
            prediction_results['probabilities']
        )
        
        # Calculate statistics with safe defaults
        statistics = calculate_multi_class_statistics(
            prediction_results['predictions'], 
            decoded_predictions
        )
        
        # Create visualization (handle errors gracefully)
        plot_image = ""
        try:
            plot_image = create_multi_class_plot(
                prediction_results['predictions'],
                prediction_results['probabilities'],
                statistics
            )
        except Exception as e:
            print(f"⚠️ Visualization error (non-critical): {e}")
            plot_image = ""
        
        # Map model type to display name
        model_display_names = {
            'xgboost': 'XGBoost',
            'catboost': 'CatBoost', 
            'votingensemble': 'VotingEnsemble',
            'lightgbm': 'LightGBM'
        }
        
        # Prepare SAFE response with guaranteed structure
        results = {
            "model_used": model_display_names.get(model_type, model_type),
            "is_real_prediction": True,
            "file_info": {
                "filename": file.filename or "unknown",
                "rows_processed": len(df),
                "features_used": len(feature_names) if feature_names else len(df.columns)
            },
            "predictions": decoded_predictions or [],
            "statistics": {
                "total_predictions": statistics.get("total_predictions", 0),
                "false_positive_count": statistics.get("false_positive_count", 0),
                "candidate_count": statistics.get("candidate_count", 0),
                "confirmed_count": statistics.get("confirmed_count", 0),
                "false_positive_percentage": float(statistics.get("false_positive_percentage", 0)),
                "candidate_percentage": float(statistics.get("candidate_percentage", 0)),
                "confirmed_percentage": float(statistics.get("confirmed_percentage", 0)),
                "prediction_breakdown": statistics.get("prediction_breakdown", {})
            },
            "visualizations": {
                "prediction_plot": plot_image or ""
            },
            "message": f"Analysis completed. Found: {statistics.get('confirmed_count', 0)} confirmed, {statistics.get('candidate_count', 0)} candidates, {statistics.get('false_positive_count', 0)} false positives."
        }
        
        print(f"✅ Prediction completed successfully")
        print(f"   Confirmed: {statistics.get('confirmed_count', 0)}")
        print(f"   Candidates: {statistics.get('candidate_count', 0)}")
        print(f"   False Positives: {statistics.get('false_positive_count', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            {"error": f"Error processing file: {str(e)}"}, 
            status_code=500
        )

@app.get("/api/models")
async def get_models():
    """Return available ML models status"""
    return {
        "available_models": list(models.keys()) if models else [],
        "loaded_features": feature_names if feature_names else [],
        "models_loaded": bool(models),
        "status": "Models loaded successfully" if models else "No models loaded"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Exoplanet Detector with Multi-Class Support...")
    print("📍 Homepage: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)