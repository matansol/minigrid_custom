// Connect to the Socket.IO server with polling+WebSocket fallback for local testing
const socket = io({
    transports: ["polling", "websocket"],  // Allow polling fallback for local/dev
    timeout: 20000,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    maxReconnectionAttempts: 5,
    forceNew: false,  // Don't force new connection unless needed
    path: "/socket.io",  // Explicit path
    autoConnect: false  // Don't auto-connect, we'll connect manually
});

// Connection management
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
            if (connectionAttempts < 5) {  // Fewer attempts since WebSocket fails faster
                setTimeout(() => connectSocket(), 1000 * connectionAttempts);
            } else {
                console.log('Max connection attempts reached');
                // Could show user an error message here
            }
        }
    }, 5000);  // 5 second timeout for WebSocket
}

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
let episodesCompleted = 0;
let roundScores = [];
let episodeNum = 1;

// Event Listeners

// --- PROLIFIC ID HANDLING ---
function getProlificIdOrRandom() {
    const params = new URLSearchParams(window.location.search);
    let prolificId = params.get('prolificID');
    if (prolificId && prolificId.trim() !== '') {
        return prolificId;
    } else {
        // Generate a random number between 1 and 100
        return Math.floor(Math.random() * 100) + 1;
    }
}
const prolificID = getProlificIdOrRandom();

// // --- FINAL STEP HANDLING ---
// function getFinalStepParameter() {
//     const params = new URLSearchParams(window.location.search);
//     let finalStep = 1 //params.get('finalStep');
//     return finalStep; // === '0' ? 0 : 1;
// }
const finalStep = 1;//getFinalStepParameter();  

startTutorialButton.addEventListener('click', () => {
    showLoading();
    connectSocket();  // Connect to socket before starting game
    socket.emit('start_game', { playerName: prolificID, finalStep: 1 });
});

// Keyboard controls
document.addEventListener('keydown', (event) => {
    if (!gamePage.classList.contains('active')) return;

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
        socket.emit('send_action', action);
    }
});

// Socket.IO event handlers for polling+WebSocket fallback
socket.on('connect', () => {
    console.log('WebSocket connected to server with socket ID:', socket.id);
    isConnecting = false;
    connectionAttempts = 0;  // Reset connection attempts on successful connection
});

socket.on('connect_error', (error) => {
    console.log('WebSocket connection error:', error);
    isConnecting = false;
});

socket.on('disconnect', (reason) => {
    console.log('WebSocket disconnected from server. Reason:', reason);
    if (reason === 'transport error') {
        console.log('Attempting WebSocket reconnection...');
        setTimeout(() => connectSocket(), 1000);
    }
});

socket.on('game_update', (data) => {
    updateGameState(data);
});

socket.on('episode_finished', (data) => {
    updateGameState(data);
    episodeNum++;
    if (data.score !== undefined) {
        roundScores.push(data.score);
    }
    
    // Handle finalStep case - finish after just 1 episode
    if (finalStep === 1) {
        gamePage.classList.remove('active');
        finishPage.classList.add('active');
        
        // Calculate score level and update confirmation number
        const score = data.score || 0;
        let scoreLevel = 1;
        if (score > 13) scoreLevel = 3;
        else if (score > 11) scoreLevel = 2;
        else scoreLevel = 1;

        
        const confirmationNumber = `232${scoreLevel}`;
        const confirmationElement = document.getElementById('confirmation-number');
        if (confirmationElement) {
            confirmationElement.textContent = confirmationNumber;
        }
        
        // Populate the scores list on the finish page
        if (scoreList) {
            scoreList.innerHTML = '';
            const li = document.createElement('li');
            li.textContent = `Final Score: ${score}`;
            li.style.listStyleType = 'none';
            scoreList.appendChild(li);
        }
        return;
    }
    
});

socket.on('error', (data) => {
    alert(`Error: ${data.error}`);
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
            roundNumberElement.textContent = 1;//currentEpisode;
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