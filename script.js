// --- Global variables ---
let model;
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predict-button');
const clearButton = document.getElementById('clear-button');
const predictionResult = document.getElementById('prediction-result');

let isDrawing = false;

// --- Load the model ---
async function loadModel() {
    console.log("Loading model...");
    // The path must match the folder structure
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Model loaded successfully!");
    // Enable buttons once model is loaded
    predictButton.disabled = false;
    clearButton.disabled = false;
    predictButton.innerText = "Predict";
}

// --- Canvas drawing functions ---
function startDrawing(e) {
    isDrawing = true;
    draw(e); // Start drawing immediately
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath(); // Reset the path to start a new line
}

function draw(e) {
    if (!isDrawing) return;

    // Set drawing properties
    ctx.lineWidth = 15; // A good thickness for a 280x280 canvas
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white'; // Draw in white on black background

    // Get mouse/touch position
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// --- Button functions ---
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictionResult.innerText = '';
}

async function predictDigit() {
    // tf.tidy helps manage memory by cleaning up intermediate tensors
    const prediction = tf.tidy(() => {
        // 1. Get image from canvas
        let img = tf.browser.fromPixels(canvas, 1); // 1 channel for grayscale

        // 2. Resize to 28x28
        img = tf.image.resizeBilinear(img, [28, 28]);
        
        // 3. Reshape to the format our model expects: [1, 28, 28, 1]
        // (1 sample, 28x28 pixels, 1 color channel)
        const reshapedImg = img.reshape([1, 28, 28, 1]);

        // 4. Cast to float32
        const castedImg = tf.cast(reshapedImg, 'float32');

        // 5. Make the prediction
        const output = model.predict(castedImg);

        // 6. Get the prediction with the highest score
        return output.argMax(1).dataSync()[0];
    });

    // Display the prediction
    predictionResult.innerText = prediction;
}


// --- Event Listeners ---
window.onload = () => {
    // Disable buttons until model is loaded
    predictButton.disabled = true;
    predictButton.innerText = "Loading Model...";
    
    loadModel();

    // Mouse events for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', stopDrawing); // Stop if mouse leaves canvas

    // Button events
    predictButton.addEventListener('click', predictDigit);
    clearButton.addEventListener('click', clearCanvas);
};