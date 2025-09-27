/**
 * EXOPLANET DETECTOR - PREDICTION SCRIPT
 * Handles file upload, preview, model selection, and API communication
 */

// ===== DOM ELEMENTS =====
// Theory: We cache DOM elements for better performance and readability
const elements = {
    // Form elements
    predictionForm: document.getElementById('predictionForm'),
    modelSelect: document.getElementById('modelSelect'),
    csvFile: document.getElementById('csvFile'),
    predictBtn: document.getElementById('predictBtn'),
    confirmPredict: document.getElementById('confirmPredict'),
    
    // UI sections
    previewSection: document.getElementById('previewSection'),
    csvPreview: document.getElementById('csvPreview'),
    loadingSpinner: document.getElementById('loadingSpinner'),
    resultsSection: document.getElementById('resultsSection'),
    resultsContent: document.getElementById('resultsContent')
};

// ===== CONFIGURATION =====
// Theory: Constants for easy configuration and maintenance
const config = {
    maxFileSize: 10 * 1024 * 1024, // 10MB in bytes
    apiEndpoints: {
        predict: '/api/predict',
        test: '/api/test'
    },
    allowedFileTypes: ['.csv', 'text/csv']
};

// ===== APPLICATION INITIALIZATION =====
// Theory: Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Exoplanet Detector initialized');
    initializeEventListeners();
    validateForm();
});

/**
 * Initialize all event listeners
 * Theory: Event listeners handle user interactions
 */
function initializeEventListeners() {
    // Form submission
    elements.predictionForm.addEventListener('submit', handleFormSubmit);
    
    // File input change - trigger preview
    elements.csvFile.addEventListener('change', handleFileSelect);
    
    // Model selection change - validate form
    elements.modelSelect.addEventListener('change', validateForm);
    
    // Confirm prediction button
    elements.confirmPredict.addEventListener('click', handlePrediction);
    
    console.log('✅ Event listeners initialized');
}

/**
 * Handle form submission
 * Theory: Prevent default form behavior, show preview instead of immediate submission
 */
function handleFormSubmit(event) {
    event.preventDefault(); // Theory: Prevent page reload
    console.log('📝 Form submitted');
    
    if (validateFile()) {
        showFilePreview();
    }
}

/**
 * Handle file selection
 * Theory: Validate file immediately when user selects it
 */
function handleFileSelect() {
    console.log('📁 File selected');
    validateFile();
    validateForm();
}

/**
 * Validate selected file
 * Theory: Client-side validation before sending to server
 */
function validateFile() {
    const file = elements.csvFile.files[0];
    
    if (!file) {
        showError('Please select a CSV file');
        return false;
    }
    
    // Check file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showError('Please select a valid CSV file');
        elements.csvFile.value = ''; // Clear invalid selection
        return false;
    }
    
    // Check file size
    if (file.size > config.maxFileSize) {
        showError('File size too large. Maximum 10MB allowed.');
        elements.csvFile.value = '';
        return false;
    }
    
    console.log('✅ File validation passed');
    return true;
}

/**
 * Validate entire form
 * Theory: Enable/disable buttons based on form validity
 */
function validateForm() {
    const fileValid = elements.csvFile.files.length > 0;
    const modelValid = elements.modelSelect.value !== '';
    const formValid = fileValid && modelValid;
    
    elements.predictBtn.disabled = !formValid;
    
    // Theory: Visual feedback for disabled state
    if (formValid) {
        elements.predictBtn.style.opacity = '1';
        elements.predictBtn.style.cursor = 'pointer';
    } else {
        elements.predictBtn.style.opacity = '0.6';
        elements.predictBtn.style.cursor = 'not-allowed';
    }
    
    return formValid;
}

/**
 * Show file preview
 * Theory: Use FileReader API to read and display CSV content
 */
function showFilePreview() {
    const file = elements.csvFile.files[0];
    
    if (!file) return;
    
    const reader = new FileReader();
    
    // Theory: FileReader events for handling file reading
    reader.onload = function(e) {
        const csvContent = e.target.result;
        displayCSVPreview(csvContent);
        elements.previewSection.style.display = 'block';
        
        // Smooth scroll to preview
        elements.previewSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    };
    
    reader.onerror = function() {
        showError('Error reading file');
    };
    
    // Theory: readAsText reads file as plain text
    reader.readAsText(file);
}

/**
 * Display CSV preview in a formatted way
 * Theory: Parse CSV and create a readable table preview
 */
function displayCSVPreview(csvContent) {
    const lines = csvContent.split('\n').slice(0, 10); // Show first 10 lines
    let previewHTML = '<div class="preview-table">';
    
    lines.forEach((line, index) => {
        const cells = line.split(',').map(cell => 
            `<span class="csv-cell">${cell.trim()}</span>`
        ).join('<span class="csv-separator">, </span>');
        
        previewHTML += `<div class="csv-line">${cells}</div>`;
        
        // Theory: Add line numbers for better readability
        if (index === 0) {
            previewHTML += '<div class="csv-divider">...</div>';
        }
    });
    
    previewHTML += '<div class="preview-info">Showing first 10 rows</div></div>';
    elements.csvPreview.innerHTML = previewHTML;
}

/**
 * Handle prediction request
 * Theory: Main function that coordinates the prediction flow
 */
async function handlePrediction() {
    console.log('🎯 Starting prediction process');
    
    if (!validateForm()) {
        showError('Please complete the form before predicting');
        return;
    }
    
    try {
        // Theory: Show loading state for better UX
        setLoadingState(true);
        hideResults();
        
        // Send prediction request
        const result = await sendPredictionRequest();
        
        // Display results
        displayPredictionResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        handlePredictionError(error);
    } finally {
        setLoadingState(false);
    }
}

/**
 * Send prediction request to backend
 * Theory: Use Fetch API to communicate with FastAPI
 */
async function sendPredictionRequest() {
    const file = elements.csvFile.files[0];
    const modelType = elements.modelSelect.value;
    
    // Theory: FormData allows file upload via HTTP
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    
    console.log(`📤 Sending request for ${modelType} model`);
    
    const response = await fetch(config.apiEndpoints.predict, {
        method: 'POST',
        body: formData
        // Theory: Don't set Content-Type header for FormData - browser handles it
    });
    
    // Theory: Check HTTP status code
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText}`);
    }
    
    // Theory: Parse JSON response
    return await response.json();
}

/**
 * Display prediction results
 * Theory: Format and show results from the API response
 */
function displayPredictionResults(result) {
    console.log('📊 Displaying results:', result);
    
    let resultsHTML = `
        <div class="result-header">
            <h3 class="text-blue">Model Used: ${result.model_used || 'Unknown'}</h3>
            <p class="timestamp">Analysis completed: ${new Date().toLocaleString()}</p>
            <div class="file-info">
                <p><strong>File:</strong> ${result.file_info.filename}</p>
                <p><strong>Rows processed:</strong> ${result.file_info.rows_processed}</p>
                <p><strong>Features used:</strong> ${result.file_info.features_used}</p>
            </div>
        </div>
    `;
    
    // Add statistics with SAFE number handling
    if (result.statistics) {
        const stats = result.statistics;
        
        resultsHTML += `
            <div class="statistics-card">
                <h4 class="text-blue">📈 Prediction Statistics</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Total Predictions:</span>
                        <span class="stat-value">${stats.total_predictions || 0}</span>
                    </div>
                    <div class="stat-item" style="border-left-color: #44ff44;">
                        <span class="stat-label">Confirmed Exoplanets:</span>
                        <span class="stat-value" style="color: #44ff44;">${stats.confirmed_count || 0}</span>
                    </div>
                    <div class="stat-item" style="border-left-color: #ffa500;">
                        <span class="stat-label">Candidates:</span>
                        <span class="stat-value" style="color: #ffa500;">${stats.candidate_count || 0}</span>
                    </div>
                    <div class="stat-item" style="border-left-color: #ff4444;">
                        <span class="stat-label">False Positives:</span>
                        <span class="stat-value" style="color: #ff4444;">${stats.false_positive_count || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Confirmed %:</span>
                        <span class="stat-value">${(stats.confirmed_percentage || 0).toFixed(1)}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Candidate %:</span>
                        <span class="stat-value">${(stats.candidate_percentage || 0).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add visualizations
    if (result.visualizations && result.visualizations.prediction_plot) {
        resultsHTML += `
            <div class="visualization-section">
                <h3 class="text-blue">📊 Analysis Visualizations</h3>
                <div class="plot-container">
                    <img src="${result.visualizations.prediction_plot}" alt="Prediction Analysis" 
                         style="max-width: 100%; border: 1px solid #333; border-radius: 8px;">
                </div>
            </div>
        `;
    }
    
    // Add predictions table
    if (result.predictions && Array.isArray(result.predictions)) {
        resultsHTML += createMultiClassPredictionsTable(result.predictions);
    }
    
    // Add message
    if (result.message) {
        resultsHTML += `
            <div class="result-message">
                <h4 class="text-blue">💡 Analysis Summary</h4>
                <p>${result.message}</p>
            </div>
        `;
    }
    
    elements.resultsContent.innerHTML = resultsHTML;
    elements.resultsSection.classList.add('active');
    
    // Smooth scroll to results
    elements.resultsSection.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function createMultiClassPredictionsTable(predictions) {
    if (!predictions || predictions.length === 0) {
        return '<p>No predictions available</p>';
    }
    
    let tableHTML = `
        <div class="predictions-section">
            <h4 class="text-blue">📋 Detailed Predictions</h4>
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Classification</th>
                        <th>Confidence</th>
                        <th>Probabilities</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    predictions.forEach((pred, index) => {
        const confidence = pred.confidence ? (pred.confidence * 100).toFixed(1) + '%' : 'N/A';
        
        // Create probability bars if available
        let probabilityHTML = 'N/A';
        if (pred.probabilities) {
            const probs = pred.probabilities;
            probabilityHTML = `
                <div class="probability-bars">
                    <div class="prob-bar">
                        <span>FP: ${(probs.false_positive * 100).toFixed(1)}%</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width: ${probs.false_positive * 100}%; background: #ff4444;"></div>
                        </div>
                    </div>
                    <div class="prob-bar">
                        <span>Cand: ${(probs.candidate * 100).toFixed(1)}%</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width: ${probs.candidate * 100}%; background: #ffa500;"></div>
                        </div>
                    </div>
                    <div class="prob-bar">
                        <span>Conf: ${(probs.confirmed * 100).toFixed(1)}%</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width: ${probs.confirmed * 100}%; background: #44ff44;"></div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        tableHTML += `
            <tr>
                <td>${pred.row}</td>
                <td style="color: ${pred.color || '#ffffff'}">${pred.prediction}</td>
                <td>${confidence}</td>
                <td>${probabilityHTML}</td>
            </tr>
        `;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    return tableHTML;
}
 // Visulizations section
function displayVisualizations(visualizations) {
    let visualizationHTML = '<div class="visualization-section">';
    visualizationHTML += '<h3 class="text-blue">📊 Analysis Visualizations</h3>';
    
    if (visualizations.prediction_plot) {
        visualizationHTML += `
            <div class="plot-container">
                <h4>Prediction Analysis</h4>
                <img src="${visualizations.prediction_plot}" alt="Prediction Analysis" 
                     style="max-width: 100%; border: 1px solid #333; border-radius: 8px;">
            </div>
        `;
    }
    
    if (visualizations.feature_distribution) {
        visualizationHTML += `
            <div class="plot-container">
                <h4>Feature Distribution</h4>
                <img src="${visualizations.feature_distribution}" alt="Feature Distribution" 
                     style="max-width: 100%; border: 1px solid #333; border-radius: 8px;">
            </div>
        `;
    }
    
    visualizationHTML += '</div>';
    return visualizationHTML;
}

/**
 * Create predictions table
 * Theory: Generate HTML table from prediction data
 */
function createPredictionsTable(predictions) {
    // Theory: Handle both array and object formats
    const predictionArray = Array.isArray(predictions) ? predictions : [predictions];
    
    if (predictionArray.length === 0) {
        return '<p>No predictions available</p>';
    }
    
    let tableHTML = `
        <div class="predictions-section">
            <h4 class="text-blue">📈 Exoplanet Predictions</h4>
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Theory: Loop through predictions and create table rows
    predictionArray.forEach((pred, index) => {
        const confidence = pred.confidence ? `${(pred.confidence * 100).toFixed(1)}%` : 'N/A';
        const predictionText = pred.prediction === 1 ? '🌍 Exoplanet Detected' : '⭐ Star Variation';
        
        tableHTML += `
            <tr>
                <td>${index + 1}</td>
                <td>${predictionText}</td>
                <td>${confidence}</td>
            </tr>
        `;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    return tableHTML;
}

/**
 * Create probabilities section
 * Theory: Display probability distributions if available
 */
function createProbabilitiesSection(probabilities) {
    return `
        <div class="probabilities-section">
            <h4 class="text-blue">📊 Probability Distribution</h4>
            <pre class="probabilities-json">${JSON.stringify(probabilities, null, 2)}</pre>
        </div>
    `;
}

/**
 * Create statistics section
 * Theory: Show analysis statistics
 */
function createStatisticsSection(statistics) {
    let statsHTML = `
        <div class="statistics-section">
            <h4 class="text-blue">📐 Analysis Statistics</h4>
            <div class="stats-grid">
    `;
    
    // Theory: Loop through statistics object
    for (const [key, value] of Object.entries(statistics)) {
        statsHTML += `
            <div class="stat-item">
                <span class="stat-label">${key}:</span>
                <span class="stat-value">${value}</span>
            </div>
        `;
    }
    
    statsHTML += `
            </div>
        </div>
    `;
    
    return statsHTML;
}

/**
 * Handle prediction errors
 * Theory: Centralized error handling with user-friendly messages
 */
function handlePredictionError(error) {
    let errorMessage = 'An unexpected error occurred during analysis.';
    
    // Theory: Different error types need different handling
    if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
        errorMessage = 'Network error: Please check your connection and try again.';
    } else if (error.message.includes('413')) {
        errorMessage = 'File too large: Please select a smaller CSV file.';
    } else if (error.message.includes('415')) {
        errorMessage = 'Invalid file type: Please upload a valid CSV file.';
    } else {
        errorMessage = error.message;
    }
    
    elements.resultsContent.innerHTML = `
        <div class="error-message">
            <h4 class="text-red">❌ Analysis Error</h4>
            <p>${errorMessage}</p>
            <p><small>Please try again or contact support if the problem persists.</small></p>
        </div>
    `;
    
    elements.resultsSection.classList.add('active');
}

/**
 * Set loading state
 * Theory: Show/hide loading spinner and manage button states
 */
function setLoadingState(isLoading) {
    if (isLoading) {
        elements.loadingSpinner.classList.add('active');
        elements.predictBtn.disabled = true;
        elements.confirmPredict.disabled = true;
        elements.predictBtn.innerHTML = '⏳ Analyzing...';
    } else {
        elements.loadingSpinner.classList.remove('active');
        elements.predictBtn.disabled = false;
        elements.confirmPredict.disabled = false;
        elements.predictBtn.innerHTML = '🚀 Analyze Data';
    }
}

/**
 * Hide results section
 * Theory: Reset results when starting new prediction
 */
function hideResults() {
    elements.resultsSection.classList.remove('active');
    elements.resultsContent.innerHTML = '';
}

/**
 * Show temporary error message
 * Theory: Non-intrusive error notifications
 */
function showError(message) {
    // Theory: Create temporary error message element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'temp-error';
    errorDiv.innerHTML = `
        <div style="
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff4444;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            z-index: 10000;
            max-width: 300px;
            animation: slideIn 0.3s ease;
        ">
            ${message}
        </div>
    `;
    
    document.body.appendChild(errorDiv);
    
    // Theory: Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);
}

// Theory: Export for testing if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateFile,
        displayPredictionResults,
        handlePrediction
    };
}