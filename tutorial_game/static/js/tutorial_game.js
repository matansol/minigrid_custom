// Connect to the Socket.IO server
const socket = io();

// DOM Elements
const coverPage = document.getElementById('cover-page');
const welcomePage = document.getElementById('welcome-page');
const gamePage = document.getElementById('game-page');
const finishPage = document.getElementById('finish-page');
const coverStartButton = document.getElementById('cover-start-button');
const startTutorialButton = document.getElementById('start-tutorial');
const playerNameInput = document.getElementById('player-name');
const gameImage = document.getElementById('game-image');
const scoreElement = document.getElementById('score');
const episodeElement = document.getElementById('episode');
const stepsElement = document.getElementById('steps');
const rewardElement = document.getElementById('reward');
const nextEpisodeButton = document.getElementById('next-episode');
const loadingOverlay = document.getElementById('loading-overlay');
const finishButtonContainer = document.getElementById('finish-button-container');

// Game state
let currentEpisode = 1;
let currentScore = 0;
let currentSteps = 0;
let episodesCompleted = 0;

// Event Listeners
coverStartButton.addEventListener('click', () => {
    coverPage.classList.remove('active');
    welcomePage.classList.add('active');
});

startTutorialButton.addEventListener('click', () => {
    const playerName = playerNameInput.value.trim();
    if (playerName) {
        showLoading();
        socket.emit('start_game', { playerName });
    } else {
        alert('Please enter your name to continue.');
    }
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

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('game_update', (data) => {
    updateGameState(data);
});

socket.on('episode_finished', (data) => {
    updateGameState(data);
    episodesCompleted++;
    
    if (episodesCompleted >= 2) {
        // Create and show the finish tutorial button only if it doesn't already exist
        if (!document.getElementById('finish-tutorial-btn')) {
            const finishButton = document.createElement('button');
            finishButton.id = 'finish-tutorial-btn';
            finishButton.textContent = 'Finish Tutorial';
            finishButton.style.padding = '10px 20px';
            finishButton.style.fontSize = '16px';
            finishButton.style.backgroundColor = '#4CAF50'; // Green button
            finishButton.style.color = 'white';
            finishButton.style.border = 'none';
            finishButton.style.borderRadius = '5px';
            finishButton.style.cursor = 'pointer';
            finishButton.addEventListener('click', () => {
                gamePage.classList.remove('active');
                finishPage.classList.add('active');
            });

            // Ensure the finish button container is visible and positioned correctly
            finishButtonContainer.appendChild(finishButton);
        }
    } else {
        // Remove the button if not enough episodes completed
        const existingBtn = document.getElementById('finish-tutorial-btn');
        if (existingBtn) {
            finishButtonContainer.removeChild(existingBtn);
        }
    }
    
    alert(`Round finished! Your score: ${data.score}`);
    socket.emit('next_episode');
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
        episodeElement.textContent = currentEpisode;
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