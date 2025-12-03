// Update gesture data every 500ms
let gestureHistory = [];
let maxHistoryItems = 10;

function updateGestureData() {
    fetch('/gesture_data')
        .then(response => response.json())
        .then(data => {
            // Update current gesture
            updateCurrentGesture(data.gesture, data.confidence);
            
            // Update volume control
            updateVolumeControl(data.volume);
            
            // Update brightness control
            updateBrightnessControl(data.brightness);
            
            // Update history
            updateHistory(data.gesture);
        })
        .catch(error => {
            console.error('Error fetching gesture data:', error);
        });
}

function updateCurrentGesture(gesture, confidence) {
    const gestureNameElement = document.getElementById('gestureName');
    const gestureIconElement = document.querySelector('.gesture-icon');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    
    // Update gesture name
    gestureNameElement.textContent = gesture;
    
    // Update gesture icon based on detected gesture
    const gestureIcons = {
        'One Finger': '1️⃣',
        'Two Fingers': '2️⃣',
        'Three Fingers': '3️⃣',
        'Four Fingers': '4️⃣',
        'Five Fingers': '5️⃣',
        'Peace': '✌️',
        'Thumbs Up': '👍',
        'Open Palm': '🖐️',
        'Fist': '✊',
        'Pointing': '☝️',
        'OK Sign': '👌',
        'Rock': '🤘',
        'Volume Control': '🔊',
        'Brightness Control': '💡',
        'No Hand Detected': '🚫',
        'None': '👋'
    };
    
    // Find matching icon
    let icon = '👋';
    for (const [key, value] of Object.entries(gestureIcons)) {
        if (gesture.includes(key)) {
            icon = value;
            break;
        }
    }
    gestureIconElement.textContent = icon;
    
    // Update confidence
    confidenceValue.textContent = confidence + '%';
    confidenceFill.style.width = confidence + '%';
    
    // Add animation when gesture changes
    if (confidence > 0) {
        gestureIconElement.style.animation = 'none';
        setTimeout(() => {
            gestureIconElement.style.animation = 'bounce 2s ease-in-out infinite';
        }, 10);
    }
}

function updateVolumeControl(volume) {
    const volumeValue = document.getElementById('volumeValue');
    const volumeFill = document.getElementById('volumeFill');
    
    volumeValue.textContent = volume + '%';
    volumeFill.style.width = volume + '%';
}

function updateBrightnessControl(brightness) {
    const brightnessValue = document.getElementById('brightnessValue');
    const brightnessFill = document.getElementById('brightnessFill');
    
    brightnessValue.textContent = brightness + '%';
    brightnessFill.style.width = brightness + '%';
}

function updateHistory(gesture) {
    if (gesture === 'None' || gesture === 'No Hand Detected') {
        return;
    }
    
    // Add to history if it's a new gesture or different from the last one
    if (gestureHistory.length === 0 || gestureHistory[gestureHistory.length - 1] !== gesture) {
        gestureHistory.push(gesture);
        
        // Limit history size
        if (gestureHistory.length > maxHistoryItems) {
            gestureHistory.shift();
        }
        
        // Update history display
        const historyList = document.getElementById('historyList');
        historyList.innerHTML = '';
        
        // Display in reverse order (most recent first)
        for (let i = gestureHistory.length - 1; i >= 0; i--) {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.textContent = gestureHistory[i];
            historyList.appendChild(historyItem);
        }
    }
}

// Add visual feedback for gesture items
document.querySelectorAll('.gesture-item').forEach(item => {
    item.addEventListener('click', function() {
        this.style.transform = 'scale(1.1)';
        setTimeout(() => {
            this.style.transform = 'scale(1)';
        }, 200);
    });
});

// Smooth animations on page load
window.addEventListener('load', () => {
    document.querySelectorAll('.card').forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// Status indicator animation
const statusDot = document.querySelector('.status-dot');
let isConnected = true;

function checkConnection() {
    fetch('/gesture_data')
        .then(() => {
            if (!isConnected) {
                statusDot.style.background = '#10b981';
                isConnected = true;
            }
        })
        .catch(() => {
            if (isConnected) {
                statusDot.style.background = '#ef4444';
                isConnected = false;
            }
        });
}

// Initialize updates
setInterval(updateGestureData, 500);
setInterval(checkConnection, 3000);

// Initial load
updateGestureData();

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') {
        location.reload();
    }
});

// Display welcome message
console.log('%c🚀 Hand Gesture Control System Initialized!', 'color: #6366f1; font-size: 20px; font-weight: bold;');
console.log('%c👋 Show your hand gestures to the camera', 'color: #8b5cf6; font-size: 14px;');
console.log('%cPress R to reload the page', 'color: #94a3b8; font-size: 12px;');
