// Connect to the Socket.IO server with polling+WebSocket fallback for local testing
const socket = io({
    transports: ["polling", "websocket"],  // Allow polling fallback for local/dev
    // Note: in production you can prefer websocket-only by using ["websocket"] and upgrade=false
    timeout: 20000,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    maxReconnectionAttempts: 5,
    forceNew: false,  // Don't force new connection unless needed
    path: "/socket.io",  // Explicit path
    autoConnect: false  // Don't auto-connect, we'll connect manually
});


// DOM Elements
const welcomePage = document.getElementById('welcome-page');
const gamePage = document.getElementById('game-page');
const finishPage = document.getElementById('finish-page');
const scoreList = document.getElementById('score-list');
const startTutorialButton = document.getElementById('start-tutorial');
const playerNameInput = document.getElementById('player-name');
const gameImage = document.getElementById('game-image');
const scoreElement = document.getElementById('score');
// const episodeElement = document.getElementById('episode');
const stepsElement = document.getElementById('steps');
const rewardElement = document.getElementById('reward');
const nextEpisodeButton = document.getElementById('next-episode');
const loadingOverlay = document.getElementById('loading-overlay');
const finishButtonContainer = document.getElementById('finish-button-container');
const roundNumberElement = document.getElementById('round-number');

// Game state
let currentEpisode = 1;
let currentScore = 0;
let currentSteps = 0;
let episodeNum = 1;
let roundScores = [];
let connectionAttempts = 0;
let isConnecting = false;

// Connection management for WebSocket-only mode
function connectSocket() {
    if (isConnecting || socket.connected) {
        return;
    }
    
    isConnecting = true;
    connectionAttempts++;
    console.log(`Attempting WebSocket connection (attempt ${connectionAttempts})...`);
    
    socket.connect();
    
    // Set a timeout for connection attempt (shorter for WebSocket)
    setTimeout(() => {
        if (!socket.connected && isConnecting) {
            console.log('WebSocket connection attempt timed out');
            isConnecting = false;
            if (connectionAttempts < 3) {  // Fewer attempts since WebSocket fails faster
                setTimeout(() => connectSocket(), 1000 * connectionAttempts);
            } else {
                alert('Unable to establish WebSocket connection to server. Please check your internet connection and refresh the page.');
            }
        }
    }, 10000);  // Shorter timeout for WebSocket
}

// Initialize connection when page loads
window.addEventListener('load', () => {
    connectSocket();
});

// Event Listeners

// --- PROLIFIC ID HANDLING ---
function getProlificIdOrRandom() {
    const params = new URLSearchParams(window.location.search);
    let prolificId = params.get('prolificID');
    if (prolificId && prolificId.trim() !== '') {
        return prolificId;
    } else {
        // Generate a more unique random identifier
        const timestamp = Date.now();
        const random = Math.floor(Math.random() * 10000);
        return `user_${timestamp}_${random}`;
    }
}
const prolificID = getProlificIdOrRandom();

startTutorialButton.addEventListener('click', () => {
    showLoading();
    socket.emit('start_game', { playerName: prolificID });
});

// Keyboard controls
document.addEventListener('keydown', (event) => {
    if (!gamePage.classList.contains('active') || !socket.connected) return;

    let action = null;
    switch (event.key) {
        case 'ArrowLeft':
            action = 'ArrowLeft';
            break;
        case 'ArrowRight':
            action = 'ArrowRight';
            break;
        case 'ArrowUp':
            action = 'ArrowUp';
            break;
        case '1':
            action = 'PageUp';
            break;
    }

    if (action) {
        console.log('Sending action:', action);
        socket.emit('send_action', action);
    }
});

// Socket.IO event handlers for WebSocket-only mode
socket.on('connect', () => {
    console.log('WebSocket connected to server with socket ID:', socket.id);
    isConnecting = false;
    connectionAttempts = 0;  // Reset connection attempts on successful connection
});

socket.on('connection_confirmed', (data) => {
    console.log('WebSocket connection confirmed:', data);
});

socket.on('disconnect', (reason) => {
    console.log('WebSocket disconnected from server. Reason:', reason);
    isConnecting = false;
    
    // Auto-reconnect for WebSocket disconnects
    if (reason === 'io server disconnect' || reason === 'transport close' || reason === 'transport error') {
        console.log('Attempting WebSocket reconnection...');
        setTimeout(() => {
            if (!socket.connected) {
                connectSocket();
            }
        }, 1000);  // Shorter delay for WebSocket reconnections
    }
});

socket.on('connect_error', (error) => {
    console.error('WebSocket connection error:', error);
    isConnecting = false;
    
    // Try to reconnect after a delay
    setTimeout(() => {
        if (!socket.connected && connectionAttempts < 5) {
            connectSocket();
        }
    }, 2000);
});

socket.on('reconnect', (attemptNumber) => {
    console.log('Reconnected to server after', attemptNumber, 'attempts');
    isConnecting = false;
    connectionAttempts = 0;
});

socket.on('reconnect_error', (error) => {
    console.error('Reconnection failed:', error);
});

socket.on('reconnect_failed', () => {
    console.error('Reconnection failed after maximum attempts');
    alert('Connection lost and could not be restored. Please refresh the page.');
});

// Enhanced error handling
socket.on('error', (data) => {
    console.error('Server error:', data);
    hideLoading();
    
    // Handle specific error types
    if (data.error && data.error.includes('Session not found')) {
        alert('Session expired. Please refresh the page and start again.');
    } else if (data.error && data.error.includes('Game not initialized')) {
        alert('Game session lost. Please refresh the page and start again.');
    } else {
        alert('Error: ' + (data.error || 'Unknown error occurred'));
    }
});

socket.on('game_update', (data) => {
    // console.log('Game update received:', data);
    hideLoading();
    updateGameState(data);
});

socket.on('episode_finished', (data) => {
    console.log('Episode finished:', data);
    hideLoading();
    updateGameState(data);
    if (data && data.episode){
        console.log("found data.episode");
        episodeNum = data.episode;
        console.log("episodeCompleted=", episodeNum);
    }
    if (data.score !== undefined) {
        roundScores.push(data.score);
    }
    
    // Handle finalStep case - finish after just 1 episode
    // if (finalStep === 1) {
    //     gamePage.classList.remove('active');
    //     finishPage.classList.add('active');
        
    //     // Calculate score level and update confirmation number
    //     const score = data.score || 0;
    //     let scoreLevel = 0;
    //     if (score > 12) scoreLevel = 4;
    //     else if (score > 8) scoreLevel = 3;
    //     else if (score > 5) scoreLevel = 2;
    //     else if (score > 1) scoreLevel = 1;
        
    //     const confirmationNumber = `232${scoreLevel}`;
    //     const confirmationElement = document.getElementById('confirmation-number');
    //     if (confirmationElement) {
    //         confirmationElement.textContent = confirmationNumber;
    //     }
        
    //     // Populate the scores list on the finish page
    //     if (scoreList) {
    //         scoreList.innerHTML = '';
    //         const li = document.createElement('li');
    //         li.textContent = `Final Score: ${score}`;
    //         li.style.listStyleType = 'none';
    //         scoreList.appendChild(li);
    //     }
    //     return;
    // }
    
    // Show finish button after 3 episodes (or adjust as needed)
    if (episodeNum >= 4) {
        if (!document.getElementById('finish-tutorial-btn')) {
            const finishButton = document.createElement('button');
            finishButton.id = 'finish-tutorial-btn';
            finishButton.textContent = 'Finish Tutorial';
            finishButton.style.padding = '14px 28px';
            finishButton.style.fontSize = '18px';
            finishButton.style.backgroundColor = '#4CAF50';
            finishButton.style.color = 'white';
            finishButton.style.border = 'none';
            finishButton.style.borderRadius = '8px';
            finishButton.style.cursor = 'pointer';
            finishButton.style.marginTop = '24px';
            finishButton.style.alignSelf = 'flex-start';
            finishButton.addEventListener('click', () => {
                gamePage.classList.remove('active');
                finishPage.classList.add('active');
                // Populate the scores list on the finish page
                if (scoreList) {
                    scoreList.innerHTML = '';
                    roundScores.forEach((score, idx) => {
                        const li = document.createElement('li');
                        li.textContent = `Round ${idx + 1}: ${score}`;
                        li.style.listStyleType = 'none';
                        scoreList.appendChild(li);
                    });
                }
                finishButton.remove();
            });
            // Place the button after the map-legend (below ledge, right to key-instructions)
            const mapLegend = document.getElementById('map-legend');
            if (mapLegend && mapLegend.parentNode) {
                mapLegend.parentNode.appendChild(finishButton);
            } else {
                document.body.appendChild(finishButton);
            }
        }
    } else {
        // Remove the button if not enough episodes completed
        const existingBtn = document.getElementById('finish-tutorial-btn');
        if (existingBtn) {
            existingBtn.remove();
        }
    }
    socket.emit('next_episode');
});

socket.on('error', (data) => {
    console.error('Server error:', data);
    alert(`Error: ${data.error}`);
    hideLoading();
});

// Helper functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function updateGameState(data) {
    if (data.image) {
        gameImage.src = `data:image/png;base64,${data.image}`;
    }
    if (data.score !== undefined) {
        currentScore = data.score;
        scoreElement.textContent = currentScore;
    }
    if (data.episode !== undefined) {
        currentEpisode = data.episode;
        // episodeElement.textContent = currentEpisode;
        if (roundNumberElement) {
            roundNumberElement.textContent = currentEpisode;
        }
    }
    if (data.step_count !== undefined) {
        currentSteps = data.step_count;
        stepsElement.textContent = currentSteps;
    }
    if (data.reward !== undefined) {
        rewardElement.textContent = data.reward.toFixed(2);
    }

    // Show game page if not already shown
    if (!gamePage.classList.contains('active')) {
        welcomePage.classList.remove('active');
        gamePage.classList.add('active');
        hideLoading();
    }
}

// Clean up connections when page is closed
window.addEventListener('beforeunload', () => {
    if (socket.connected) {
        socket.disconnect();
    }
});

// Clean up on page visibility change (mobile/tablet handling)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden, maintaining connection...');
    } else {
        console.log('Page visible again');
        if (!socket.connected) {
            console.log('Reconnecting...');
            socket.connect();
        }
    }
});