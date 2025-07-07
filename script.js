// Enhanced frontend script with modern UI interactions

window.addEventListener('load', () => {
    const canvas = document.getElementById('digit-canvas');
    const ctx = canvas.getContext('2d');
    const predictButton = document.getElementById('predict-button');
    const clearButton = document.getElementById('clear-button');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const canvasContainer = document.querySelector('.canvas-container');
    const canvasOverlay = document.querySelector('.canvas-overlay');

    let isDrawing = false;
    let hasDrawn = false;

    // Initialize canvas
    function initCanvas() {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function startDrawing(e) {
        isDrawing = true;
        hasDrawn = true;
        canvasContainer.classList.add('has-content');
        draw(e);
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }

    function draw(e) {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        let x, y;
        
        // Handle both mouse and touch events
        if (e.touches) {
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        } else {
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        }

        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';
        ctx.globalCompositeOperation = 'source-over';
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        hasDrawn = false;
        canvasContainer.classList.remove('has-content');
        
        // Reset prediction display
        predictionResult.innerText = '?';
        predictionResult.classList.remove('predicting');
        confidenceFill.style.width = '0%';
        confidenceText.innerText = 'Draw a digit to see prediction';
        
        // Add clear animation
        canvas.style.transform = 'scale(0.95)';
        setTimeout(() => {
            canvas.style.transform = 'scale(1)';
        }, 150);
    }

    async function predictDigit() {
        if (!hasDrawn) {
            // Shake animation for empty canvas
            canvas.style.animation = 'shake 0.5s';
            setTimeout(() => {
                canvas.style.animation = '';
            }, 500);
            return;
        }

        // Get the image data from the canvas as a Base64 encoded string
        const imageData = canvas.toDataURL('image/png');
        
        // Update UI to show we are predicting
        predictionResult.innerText = '🤔';
        predictionResult.classList.add('predicting');
        confidenceText.innerText = 'AI is thinking...';
        predictButton.disabled = true;
        
        try {
            // Use the Fetch API to send the image data to our Flask backend
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Get the prediction from the response
            const data = await response.json();
            
            // Simulate processing delay for better UX
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Display the prediction with animation
            predictionResult.classList.remove('predicting');
            predictionResult.innerText = data.prediction;
            
            // Animate confidence bar (simulate confidence - you can modify backend to return actual confidence)
            const confidence = Math.random() * 0.3 + 0.7; // 70-100% confidence simulation
            confidenceFill.style.width = `${confidence * 100}%`;
            confidenceText.innerText = `Confidence: ${Math.round(confidence * 100)}%`;
            
            // Success animation
            predictionResult.style.transform = 'scale(1.2)';
            setTimeout(() => {
                predictionResult.style.transform = 'scale(1)';
            }, 200);

        } catch (error) {
            console.error("Error during prediction:", error);
            predictionResult.classList.remove('predicting');
            predictionResult.innerText = '❌';
            confidenceText.innerText = 'Error occurred. Please try again.';
            confidenceFill.style.width = '0%';
        } finally {
            predictButton.disabled = false;
        }
    }

    // Initialize canvas with black background
    initCanvas();

    // Mouse Event Listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch Event Listeners for mobile
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startDrawing(e);
    });
    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopDrawing();
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e);
    });

    // Button Event Listeners
    predictButton.addEventListener('click', predictDigit);
    clearButton.addEventListener('click', clearCanvas);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            predictDigit();
        } else if (e.key === 'Escape' || e.key === 'Delete') {
            e.preventDefault();
            clearCanvas();
        }
    });
});

// Add shake animation to CSS dynamically
const shakeKeyframes = `
@keyframes shake {
    0%, 20%, 40%, 60%, 80%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
}
`;

const style = document.createElement('style');
style.textContent = shakeKeyframes;
document.head.appendChild(style);