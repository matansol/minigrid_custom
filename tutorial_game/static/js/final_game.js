// Connect to the Socket.IO server
const socket = io();

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

// --- FINAL STEP HANDLING ---
function getFinalStepParameter() {
    const params = new URLSearchParams(window.location.search);
    let finalStep = 1 //params.get('finalStep');
    return finalStep; // === '0' ? 0 : 1;
}
const finalStep = getFinalStepParameter();  

startTutorialButton.addEventListener('click', () => {
    showLoading();
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
    
    // Show finish button after 3 episodes (or adjust as needed)
    if (episodesCompleted >= 3) {
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