
// JavaScript code for the game client-side logic

document.addEventListener('DOMContentLoaded', (event) => {
    const socket = io.connect('http://' + document.domain + ':' + location.port);
    const startButton = document.getElementById('start-button');
    const ppoButton = document.getElementById('ppo-button');
    const finishGameButton = document.getElementById('finish-game-button');
    const gameContainer = document.getElementById('game-container');
    const infoContainer = document.getElementById('info-container');
    const playerNameInput = document.getElementById('player-name');
    const gameStats = document.getElementById('game-stats'); // Reference to the game stats container

    // Initially hide the PPO button
    ppoButton.style.display = 'none';

    startButton.addEventListener('click', () => {
        const playerName = playerNameInput.value;
        if (!playerName) {
            alert("Please enter your name.");
            return;
        }
        socket.emit('start_game', { playerName: playerName });
        playerNameInput.style.display = 'none';
        startButton.style.display = 'none';
        ppoButton.style.display = 'block';
        gameContainer.style.display = 'flex';
        infoContainer.style.display = 'block';
        finishGameButton.style.display = 'block';
    });

    function displayScores(scores) {
        const scoreList = document.getElementById('score-list');
        scoreList.innerHTML = ''; // Clear previous scores
        
        scores.forEach((score, index) => {
            const scoreItem = document.createElement('li');
            scoreItem.textContent = `Episode ${index + 1}: ${score}`;
            scoreList.appendChild(scoreItem);
        });

        hideGameElements(); // Hide all other elements
        gameStats.style.display = 'block'; // Show only the game stats
    }

    function hideGameElements() {
        playerNameInput.style.display = 'none';
        startButton.style.display = 'none';
        ppoButton.style.display = 'none';
        gameContainer.style.display = 'none';
        infoContainer.style.display = 'none';
        finishGameButton.style.display = 'none'; // Optionally hide this as well
    }

    finishGameButton.addEventListener('click', () => {
        socket.emit('finish_game');
    });

    socket.on('game_finished', function(data) {
        console.log("Received scores:", data.scores);  // Client-side console log for debugging
        displayScores(data.scores);
    });
    

    ppoButton.addEventListener('click', () => {
        socket.emit('ppo_action');  // Emit PPO action event
    });

    socket.on('game_update', function(data) {
        const gameImage = document.getElementById('game-image');
        gameImage.src = 'data:image/png;base64,' + data.image;
        document.getElementById('reward').innerText = data.reward; // Update reward display
        document.getElementById('score').innerText = data.score; // Update score display
        document.getElementById('last_score').innerText = data.last_score; // Update last score display
    });

    document.addEventListener('keydown', function(event) {
        const key = event.key;
        const validKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "Space", "PageUp", "PageDown", "1", "2"];
        if (validKeys.includes(key)) {
            console.log(`Sending action: ${key}`);
            socket.emit('send_action', key);
            document.getElementById('action').innerText = key; // Update action display
        }
    });
});
