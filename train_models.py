import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExoplanetModelTrainer:
    def __init__(self, data_path):
        """
        Initialize the model trainer with dataset path
        Updated for your specific exoplanet dataset
        """
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the exoplanet dataset
        Customized for your specific dataset format
        """
        print("📊 Loading exoplanet dataset...")
        
        try:
            # Load your dataset
            df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display dataset info
            print("\n📈 Dataset Overview:")
            print(f"   Columns: {list(df.columns)}")
            print(f"   First few rows:")
            print(df.head())
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\n⚠️  Missing values found:")
                for col, count in missing_values[missing_values > 0].items():
                    print(f"   {col}: {count} missing values")
            
            return self.clean_data(df)
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise e
    
    def clean_data(self, df):
        """
        Clean and prepare the exoplanet data for training
        Customized for your specific dataset
        """
        print("\n🧹 Cleaning and preparing data...")
        
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Handle missing values - fill with median for numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"   Filled missing values in {col} with median: {median_val:.4f}")
        
        # Handle the target variable 'label'
        if 'label' not in df_clean.columns:
            raise ValueError("❌ 'label' column not found in dataset!")
        
        print(f"\n🎯 Target variable 'label' distribution:")
        print(df_clean['label'].value_counts())
        
        # Convert multi-class labels to binary (Exoplanet vs Non-Exoplanet)
        # Theory: We'll treat "CONFIRMED" as exoplanets (1), others as non-exoplanets (0)
        df_clean['is_exoplanet'] = df_clean['label'].apply(
            lambda x: 1 if 'CONFIRM' in str(x).upper() else 0
        )
        
        print(f"\n🔍 Binary target distribution:")
        print(f"   Exoplanets (1): {df_clean['is_exoplanet'].sum()}")
        print(f"   Non-Exoplanets (0): {len(df_clean) - df_clean['is_exoplanet'].sum()}")
        
        # Select features for training (using your actual column names)
        # Exclude non-feature columns
        exclude_cols = ['label', 'is_exoplanet', 'source']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        print(f"\n📊 Selected features ({len(feature_cols)}): {feature_cols}")
        
        # Prepare features and target
        X = df_clean[feature_cols]
        y = df_clean['is_exoplanet']
        
        # Check for any remaining missing values
        if X.isnull().sum().sum() > 0:
            print("⚠️  Warning: Some features still have missing values after cleaning")
        
        return X, y, feature_cols
    
    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost model with optimized parameters for exoplanet data
        """
        print("🎯 Training XGBoost model...")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        return model
    
    def train_ensemble(self, X_train, y_train):
        """
        Train Ensemble model with optimized classifiers
        """
        print("🎯 Training Ensemble model...")
        
        # Define individual models optimized for your data
        estimators = [
            ('xgboost', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)),
            ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42, kernel='rbf', C=1.0))
        ]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use soft voting for probabilities
            n_jobs=-1  # Use all available cores
        )
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def train_models(self):
        """
        Main training function for exoplanet detection
        """
        print("🚀 Starting exoplanet detection model training...")
        print("="*60)
        
        # Load and preprocess data
        X, y, feature_names = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Features: {X_train.shape[1]}")
        
        # Scale features
        print("\n⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        print("\n" + "="*40)
        xgboost_model = self.train_xgboost(X_train_scaled, y_train)
        
        # Train Ensemble
        print("\n" + "="*40)
        ensemble_model = self.train_ensemble(X_train_scaled, y_train)
        
        # Evaluate models
        print("\n📊 Model Evaluation Results:")
        print("="*50)
        
        xgb_accuracy = self.evaluate_model(xgboost_model, X_test_scaled, y_test, "XGBoost")
        ensemble_accuracy = self.evaluate_model(ensemble_model, X_test_scaled, y_test, "Ensemble")
        
        # Compare models
        print("\n🏆 Model Comparison:")
        print(f"   XGBoost Accuracy: {xgb_accuracy:.4f}")
        print(f"   Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        if ensemble_accuracy > xgb_accuracy:
            print("   ✅ Ensemble model performs better!")
        else:
            print("   ✅ XGBoost model performs better!")
        
        # Store models
        self.models = {
            'xgboost': xgboost_model,
            'ensemble': ensemble_model
        }
        
        return self.models, self.scaler, feature_names
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Detailed metrics
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"   🎯 Confusion Matrix:")
        print(f"        Predicted 0  Predicted 1")
        print(f"Actual 0:    {cm[0,0]}         {cm[0,1]}")
        print(f"Actual 1:    {cm[1,0]}         {cm[1,1]}")
        
        return accuracy
    
    def save_models(self, models, scaler, feature_names):
        """
        Save trained models and preprocessing objects
        """
        print("\n💾 Saving models and preprocessing objects...")
        
        # Create models directory if it doesn't exist
        import os
        os.makedirs('ml_models', exist_ok=True)
        
        # Save individual models
        for name, model in models.items():
            filename = f'ml_models/{name}_model.pkl'
            joblib.dump(model, filename)
            print(f"   ✅ Saved {name} model to {filename}")
        
        # Save scaler
        joblib.dump(scaler, 'ml_models/scaler.pkl')
        print("   ✅ Saved feature scaler")
        
        # Save feature names
        joblib.dump(feature_names, 'ml_models/feature_names.pkl')
        print("   ✅ Saved feature names")
        
        # Save training info
        training_info = {
            'feature_names': feature_names,
            'model_types': list(models.keys()),
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(training_info, 'ml_models/training_info.pkl')
        print("   ✅ Saved training information")
        
        print("\n🎉 All models and artifacts saved successfully!")
        print("📁 Check the 'ml_models/' directory for saved files")

# Main execution
if __name__ == "__main__":
    print("🌌 Exoplanet Detection Model Trainer")
    print("="*50)
    
    # Initialize trainer with your dataset path
    dataset_path = 'data\\merged_unified_dataset.csv'  # Update this path if needed
    trainer = ExoplanetModelTrainer(dataset_path)
    
    try:
        # Train models
        models, scaler, feature_names = trainer.train_models()
        
        # Save everything
        trainer.save_models(models, scaler, feature_names)
        
        print("\n" + "="*60)
        print("🚀 EXOPLANET DETECTION MODEL TRAINING COMPLETED!")
        print("="*60)
        
        
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Check your dataset path and file format")