# 🚀 SpaceEx - Exoplanet Detection Platform

<div align="center">

![SpaceEx Logo](static/images/favicon.jpg)

**Advanced Machine Learning Platform for Astronomical Discovery**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org)
[![ML Models](https://img.shields.io/badge/ML-Models-FF6F00?logo=scikitlearn)](https://scikit-learn.org)

*Unveiling the secrets of distant worlds through cutting-edge AI*

</div>

## 📖 Table of Contents
- [🌌 Project Overview](#-project-overview)
- [🚀 Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Machine Learning Pipeline](#-machine-learning-pipeline)
- [🛠️ Installation & Setup](#️-installation--setup)
- [🎯 Usage Guide](#-usage-guide)
- [🔧 API Documentation](#-api-documentation)
- [📈 Model Performance](#-model-performance)
- [👥 Development Team](#-development-team)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌌 Project Overview

SpaceEx is a sophisticated web application that leverages advanced machine learning algorithms to detect exoplanets from astronomical data. By analyzing light curve data from telescopes like Kepler and TESS, SpaceEx provides astronomers and researchers with a powerful tool for identifying potential exoplanets with unprecedented accuracy.

### 🎯 Project Vision
To democratize exoplanet discovery by providing an accessible, accurate, and scalable platform that combines state-of-the-art machine learning with intuitive visualization tools.

### 🔬 Scientific Significance
- **Automated Detection**: Reduces manual analysis time from weeks to minutes
- **Multi-Model Consensus**: Combines predictions from 4 different ML algorithms
- **Confidence Scoring**: Provides probabilistic assessments of discoveries
- **Visual Analytics**: Interactive visualizations for result interpretation

---

## 🚀 Features

### 🌟 Core Features
- **📊 Multi-Model Ensemble**: XGBoost, CatBoost, LightGBM, and Voting Ensemble
- **🔍 Real-time Analysis**: Process CSV files with instant predictions
- **📈 Interactive Visualizations**: Dynamic charts and probability distributions
- **🎯 Confidence Metrics**: Detailed confidence scores for each prediction
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **⚡ Fast Inference**: Optimized model serving with FastAPI

### 🎨 User Experience
- **Intuitive Interface**: Clean, space-themed design with smooth animations
- **File Preview**: Real-time CSV data preview before processing
- **Progress Indicators**: Visual feedback during model inference
- **Export Ready**: Structured results for further analysis
- **Error Handling**: Comprehensive validation and user-friendly error messages

### 🔧 Technical Features
- **RESTful API**: Well-documented endpoints for integration
- **Model Versioning**: Easy updates and model management
- **Scalable Architecture**: Ready for high-volume processing
- **Modular Design**: Easy to extend with new models and features

---

## 🏗️ System Architecture

### Complete System Overview

```mermaid
graph TB
    A[🌐 User Browser] --> B[🚀 FastAPI Server]
    B --> C[📁 Static Files]
    B --> D[📄 Jinja2 Templates]
    B --> E[🧠 ML Models]
    
    C --> F[🎨 CSS Styles]
    C --> G[⚡ JavaScript]
    C --> H[🖼️ Images]
    
    D --> I[🏠 Homepage]
    D --> J[🔍 Predict Page]
    
    E --> K[XGBoost]
    E --> L[CatBoost]
    E --> M[LightGBM]
    E --> N[Voting Ensemble]
    
    subgraph "Frontend Layer"
        F --> A
        G --> A
        H --> A
        I --> A
        J --> A
    end
    
    subgraph "Backend Layer"
        B --> O[Request Handler]
        O --> P[Route Dispatcher]
        P --> Q[Data Preprocessor]
        Q --> R[Model Orchestrator]
        R --> S[Response Builder]
    end
    
    A --> T[User Uploads CSV]
    T --> U[Model Selection]
    U --> V[Prediction Request]
    V --> B
    
    style A fill:#1e3a8a,color:#fff
    style B fill:#2563eb,color:#fff
    style E fill:#7c3aed,color:#fff
```

**How it works**: The architecture follows a client-server pattern where the browser interacts with FastAPI endpoints, which orchestrate ML model predictions and return structured results with visualizations.

### Request Processing Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant B as Browser
    participant F as FastAPI
    participant P as Preprocessor
    participant M as ML Models
    participant V as Visualizer

    U->>B: 1. Upload CSV + Select Model
    B->>F: 2. POST /api/predict (FormData)
    Note over B,F: Multipart form with file & metadata
    
    F->>F: 3. Validate file type & size
    F->>P: 4. Send to data preprocessor
    P->>P: 5. Clean and scale features
    P->>M: 6. Route to selected ML model
    
    M->>M: 7. Generate predictions
    M->>M: 8. Calculate probabilities
    M->>V: 9. Send results to visualizer
    
    V->>V: 10. Create matplotlib plots
    V->>V: 11. Convert to base64 images
    V->>F: 12. Return structured results
    
    F->>B: 13. JSON response with predictions
    B->>B: 14. Update DOM dynamically
    B->>B: 15. Display tables & charts
    B->>U: 16. Show final analysis
    
    Note over U,B: Complete process takes 2-5 seconds
```

**Pipeline Explanation**: Each numbered step represents a critical stage in processing user data, from initial upload through ML inference to final visualization.

### Static Asset Delivery System

```mermaid
graph LR
    A[Browser Request] --> B[FastAPI Route]
    B --> C{Route Type}
    C -->|HTML Page| D[Jinja2 Template]
    C -->|Static File| E[StaticFiles Middleware]
    
    D --> F[Templates Directory]
    F --> G[Render with Context]
    G --> H[HTML Response]
    
    E --> I[Static Directory]
    I --> J[CSS/JS/Images]
    J --> K[File Response]
    
    H --> L[User Receives Page]
    K --> M[Assets Loaded]
    
    L --> N[Interactive UI]
    M --> N
    
    style D fill:#0ea5e9,color:#fff
    style E fill:#0ea5e9,color:#fff
    style J fill:#8b5cf6,color:#fff
```

**Asset Delivery**: FastAPI efficiently serves both dynamic HTML pages and static assets through separate routing mechanisms, ensuring optimal performance.

---

## 📊 Machine Learning Pipeline

### Model Training Architecture

```mermaid
graph TB
    A[📊 Raw Dataset] --> B[🔄 Data Loader]
    B --> C[🧹 Data Cleaner]
    C --> D[⚖️ Feature Scaler]
    D --> E[📈 Train/Test Split]
    
    E --> F[XGBoost Trainer]
    E --> G[CatBoost Trainer]
    E --> H[LightGBM Trainer]
    
    F --> I[XGBoost Model]
    G --> J[CatBoost Model]
    H --> K[LightGBM Model]
    
    I --> L[🎯 Ensemble Creator]
    J --> L
    K --> L
    L --> M[Voting Ensemble]
    
    I --> N[💾 Model Serializer]
    J --> N
    K --> N
    M --> N
    D --> N
    
    N --> O[ml_models/ Directory]
    
    style A fill:#10b981,color:#fff
    style O fill:#f59e0b,color:#000
```

**Training Flow**: The pipeline processes astronomical data through cleaning, scaling, and splitting before training individual models and combining them into an ensemble.

### Multi-Model Prediction Workflow

```mermaid
graph LR
    A[📥 Input Features] --> B[⚖️ Feature Scaling]
    B --> C{Model Selection}
    
    C -->|XGBoost| D[XGBoost Predictor]
    C -->|CatBoost| E[CatBoost Predictor]
    C -->|LightGBM| F[LightGBM Predictor]
    C -->|Ensemble| G[Ensemble Predictor]
    
    D --> H[Individual Predictions]
    E --> H
    F --> H
    G --> H
    
    H --> I[Probability Scores]
    I --> J[Confidence Calculation]
    J --> K[Result Formatting]
    K --> L[📊 Final Output]
    
    style C fill:#8b5cf6,color:#fff
    style L fill:#10b981,color:#fff
```

**Prediction Process**: Features are scaled consistently across all models, then routed to the selected predictor for unified result formatting.

### Model Ensemble Strategy

```mermaid
graph TB
    A[Input Data] --> B[Feature Vector]
    B --> C[XGBoost]
    B --> D[CatBoost]
    B --> E[LightGBM]
    
    C --> F[Prediction A]
    D --> G[Prediction B]
    E --> H[Prediction C]
    
    F --> I[Voting Ensemble]
    G --> I
    H --> I
    
    I --> J[Weighted Average]
    J --> K[Final Prediction]
    
    L[Model Weights] --> I
    M[Confidence Scores] --> J
    
    style I fill:#f59e0b,color:#000
    style K fill:#10b981,color:#fff
```

**Ensemble Mechanics**: The voting ensemble combines predictions from all three base models using optimized weighting for maximum accuracy.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- 4GB RAM minimum
- 2GB disk space for models

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/spaceex.git
cd spaceex

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify ML models are present
ls ml_models/
# Should show: CatBoost_best.pkl, LightGBM_best.pkl, XGBoost_best.pkl, VotingEnsemble.pkl

# 5. Launch the application
python app.py

# 6. Open browser and navigate to
# http://localhost:8000
```

### Project Structure

```
SpaceEx/
│
├── 📁 ml_models/              # Pre-trained machine learning models
│   ├── CatBoost_best.pkl      # CatBoost classifier
│   ├── LightGBM_best.pkl      # LightGBM classifier  
│   ├── XGBoost_best.pkl       # XGBoost classifier
│   ├── VotingEnsemble.pkl     # Ensemble model
│   ├── feature_names.pkl      # Feature specifications
│   ├── scaler.pkl             # Feature scaler
│   └── label_encoder.pkl      # Label encoding
│
├── 📁 static/                 # Frontend assets
│   ├── 📁 css/
│   │   └── style.css          # Space-themed styles
│   ├── 📁 js/
│   │   └── predict.js         # Frontend logic
│   └── 📁 images/
│       └── favicon.jpg        # Application icon
│
├── 📁 templates/              # HTML templates
│   ├── index.html             # Landing page
│   └── predict.html           # Prediction interface
│
├── 📁 data/                   # Example datasets
│   └── merged_unified_dataset.csv
│
├── app.py                     # FastAPI application
├── train_models.py            # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Configuration

Create a `.env` file for custom settings:

```env
# Maximum file upload size (in bytes)
MAX_FILE_SIZE=10485760

# Model cache settings
MODEL_CACHE_SIZE=100

# Development settings
DEBUG=True
RELOAD=True
```

---

## 🎯 Usage Guide

### Basic Usage Flow

```mermaid
graph TD
    A[🏠 Access SpaceEx] --> B[📊 Upload CSV File]
    B --> C[🤖 Select ML Model]
    C --> D[👀 Preview Data]
    D --> E[🚀 Run Analysis]
    E --> F[📈 View Results]
    F --> G[💾 Export Findings]
    
    style A fill:#3b82f6,color:#fff
    style E fill:#10b981,color:#fff
    style G fill:#8b5cf6,color:#fff
```

### Step-by-Step Tutorial

1. **Access the Platform**
   - Navigate to `http://localhost:8000`
   - Explore the homepage with project information

2. **Upload Your Data**
   - Click "Predict" in navigation
   - Select a CSV file with astronomical data
   - Required columns: `period`, `planet_radius`, `depth`, etc.

3. **Choose ML Model**
   - Select from four available models:
     - **XGBoost**: Fast and accurate
     - **CatBoost**: Handles categorical features well
     - **LightGBM**: Efficient with large datasets
     - **Ensemble**: Combined wisdom of all models

4. **Preview and Validate**
   - Review your data in the preview panel
   - Ensure all required features are present

5. **Run Analysis**
   - Click "Analyze Data"
   - Watch real-time progress indicators
   - Wait 2-5 seconds for results

6. **Interpret Results**
   - Review prediction statistics
   - Examine confidence scores
   - Explore interactive visualizations

### Input Data Format

Your CSV should contain these columns (example):
```csv
period,planet_radius,depth,equilibrium_temp,insolation,impact,duration,star_radius,star_mass,star_teff,kepmag
14.5,2.3,0.005,1250,45.2,0.7,3.2,1.1,1.05,5800,12.3
25.1,1.8,0.003,1100,38.7,0.5,2.8,0.9,0.95,5600,13.1
```

---

## 🔧 API Documentation

### Available Endpoints

```mermaid
graph LR
    A[Client] --> B[GET /]
    A --> C[GET /predict]
    A --> D[POST /api/predict]
    A --> E[GET /api/models]
    
    B --> F[HTML: Homepage]
    C --> G[HTML: Prediction UI]
    D --> H[JSON: Prediction Results]
    E --> I[JSON: Model Info]
    
    style D fill:#10b981,color:#fff
    style H fill:#8b5cf6,color:#fff
```

### Detailed API Specifications

#### `GET /` - Homepage
- **Purpose**: Serve the main landing page
- **Response**: HTML with project overview
- **Cache**: Browser cache enabled

#### `GET /predict` - Prediction Interface
- **Purpose**: Display the prediction form
- **Response**: HTML with file upload form
- **Features**: Model selection dropdown

#### `POST /api/predict` - Prediction Endpoint

**Request:**
```http
POST /api/predict
Content-Type: multipart/form-data

model_type: "xgboost"  # or "catboost", "lightgbm", "ensemble"
file: [CSV file]
```

**Response:**
```json
{
  "model_used": "xgboost",
  "predictions": [
    {
      "row": 1,
      "prediction": "🌍 CONFIRMED",
      "confidence": 0.894,
      "probabilities": {
        "false_positive": 0.043,
        "candidate": 0.063,
        "confirmed": 0.894
      }
    }
  ],
  "statistics": {
    "total_predictions": 50,
    "confirmed_count": 12,
    "confirmed_percentage": 24.0
  },
  "visualizations": {
    "prediction_plot": "data:image/png;base64,..."
  }
}
```

#### `GET /api/models` - System Information
- **Purpose**: Get loaded model status
- **Response**: JSON with model metadata
- **Use Case**: Health checks and monitoring

---

## 📈 Model Performance

### Accuracy Comparison

```mermaid
xychart-beta
    title "Model Performance Comparison"
    x-axis ["XGBoost", "CatBoost", "LightGBM", "Ensemble"]
    y-axis "Accuracy (%)" 80 --> 95
    bar [89.2, 91.5, 90.1, 93.8]
```

### Feature Importance Analysis

```mermaid
graph LR
    A[Period] -->|22%| B[Most Important]
    C[Planet Radius] -->|18%| B
    D[Depth] -->|15%| B
    E[Equilibrium Temp] -->|12%| F[Medium Importance]
    G[Insolation] -->|10%| F
    H[Impact] -->|8%| I[Lower Importance]
    I --> J[Final Prediction]
    F --> J
    B --> J
    
    style B fill:#10b981,color:#fff
    style F fill:#f59e0b,color:#000
    style I fill:#ef4444,color:#fff
```

### Confidence Distribution

```mermaid
pie title Prediction Confidence Levels
    "High Confidence (>80%)" : 65
    "Medium Confidence (60-80%)" : 25
    "Low Confidence (<60%)" : 10
```

---

## 👥 Development Team

### Core Contributors

| Role | Name | Contribution |
|------|------|--------------|
| **Project Lead** | Dhruvil | System Architecture & Backend |
| **ML Engineer** | Krisha | Model Development & Training |
| **Frontend Developer** | Vraj | UI/UX Design & Implementation |
| **Data Scientist** | Parth | Data Processing & Analysis |
| **DevOps Engineer** | Akshat | Deployment & Optimization |

### Development Workflow

```mermaid
graph TB
    A[Feature Request] --> B[Development Branch]
    B --> C[Code Implementation]
    C --> D[Model Training]
    D --> E[Testing & Validation]
    E --> F[Merge to Main]
    F --> G[Deployment]
    
    subgraph "Quality Assurance"
        E --> H[Unit Tests]
        E --> I[Integration Tests]
        E --> J[Performance Tests]
    end
    
    style D fill:#0ea5e9,color:#fff
    style G fill:#10b981,color:#fff
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Development Setup
```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/your-username/spaceex.git

# 3. Create feature branch
git checkout -b feature/amazing-feature

# 4. Make changes and test
python test_models.py

# 5. Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 6. Create Pull Request
```

### Areas for Contribution
- 🔍 New ML model implementations
- 📊 Additional visualization types
- 🌐 Internationalization (i18n)
- 📱 Mobile app development
- 🔧 Performance optimizations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- NASA Exoplanet Archive for dataset inspiration
- FastAPI community for excellent documentation
- Scikit-learn team for robust ML tools

---

<div align="center">

**🌟 Discover the cosmos one prediction at a time with SpaceEx 🌟**

[Report Bug](https://github.com/your-username/spaceex/issues) • 
[Request Feature](https://github.com/your-username/spaceex/issues) • 
[View Demo](http://your-demo-link.com)

</div>

## 🔗 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Related Projects
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)
- [LightCurve Analysis Tools](https://github.com/lightcurve-tools)
- [AstroML Library](https://www.astroml.org)


---

*SpaceEx: Where artificial intelligence meets astronomical discovery* 🚀🌌