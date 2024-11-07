document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');
    const predictButton = document.getElementById('predictButton');
    const predictionOutput = document.getElementById('predictionOutput');

    let painting = false;

    
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function startPosition(e) {
        painting = true;
        draw(e);
    }

    function endPosition() {
        painting = false;
        ctx.beginPath();
    }

    function draw(e) {
        if (!painting) return;

        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#ffffff';

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    canvas.addEventListener('mousedown', startPosition);
    canvas.addEventListener('mouseup', endPosition);
    canvas.addEventListener('mousemove', draw);

    clearButton.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillRect(0, 0, canvas.width, canvas.height); // Reset to black background
        predictionOutput.textContent = 'N/A';
    });

    predictButton.addEventListener('click', async () => {
        const dataURL = canvas.toDataURL('image/png');
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataURL }),
        });

        const data = await response.json();
        predictionOutput.textContent = data.prediction !== undefined ? data.prediction : 'Error';
    });
});
