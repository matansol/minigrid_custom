// JavaScript code for the game client-side logic

document.addEventListener('DOMContentLoaded', (event) => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = io.connect(protocol + '://' + document.domain + ':' + location.port);
    const startButton = document.getElementById('start-button');
    const ppoButton = document.getElementById('ppo-button');
    const playEpisodeButton = document.getElementById('play-episode-button'); 
    const finishGameButton = document.getElementById('finish-game-button');
    const gameContainer = document.getElementById('game-container');
    const infoContainer = document.getElementById('info-container');
    const playerNameInput = document.getElementById('player-name');
    const gameStats = document.getElementById('game-stats'); // Reference to the game stats container
    const keyInstructions = document.getElementById('key-instructions'); // Reference to key instructions

    // Initially hide the PPO button, play episode button, and key instructions
    ppoButton.style.display = 'none';
    playEpisodeButton.style.display = 'none'; // Hide the new button initially
    keyInstructions.style.display = 'none';

    startButton.addEventListener('click', () => {
        const playerName = playerNameInput.value;
        if (!playerName) {
            alert("Please enter your name.");
            return;
        }

        // Emit 'start_game' event with acknowledgment
        socket.emit('start_game', { playerName: playerName }, (response) => {
            console.log('Server acknowledged start_game:', response);
        });

        playerNameInput.style.display = 'none';
        startButton.style.display = 'none';
        ppoButton.style.display = 'block';
        playEpisodeButton.style.display = 'block'; 
        gameContainer.style.display = 'flex';
        infoContainer.style.display = 'block';
        finishGameButton.style.display = 'block';
        keyInstructions.style.display = 'block'; // Show the key instructions
    });

    playEpisodeButton.addEventListener('click', () => {
        // Emit 'play_entire_episode' event with acknowledgment
        socket.emit('play_entire_episode', (response) => {
            console.log('Server acknowledged play_entire_episode:', response);
        });
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
        playEpisodeButton.style.display = 'none'; // Hide the new button
        gameContainer.style.display = 'none';
        infoContainer.style.display = 'none';
        finishGameButton.style.display = 'none'; // Optionally hide this as well
        keyInstructions.style.display = 'none'; // Hide the key instructions
    }

    finishGameButton.addEventListener('click', () => {
        // Emit 'finish_game' event with acknowledgment
        socket.emit('finish_game', (response) => {
            console.log('Server acknowledged finish_game:', response);
        });
        keyInstructions.style.display = 'none'; // Hide key instructions after finishing the game
        playEpisodeButton.style.display = 'none'; // Hide the play episode button after finishing the game
    });

    socket.on('episode_finished', function(data) {
        // Display the path image
        const pathImage = document.getElementById('path-image');
        pathImage.src = 'data:image/png;base64,' + data.path_image;
    
        // Display the list of actions
        const actionsList = document.getElementById('actions');
        actionsList.innerHTML = '';  // Clear previous actions
        data.actions.forEach(action => {
            const actionItem = document.createElement('li');
            actionItem.textContent = action;
            actionsList.appendChild(actionItem);
        });
    
        // Show the episode info (path and actions)
        document.getElementById('episode-info').style.display = 'block';
    
        // Show the "Next Episode" button
        const nextEpisodeButton = document.getElementById('next-episode-button');
        nextEpisodeButton.style.display = 'block';
    
        nextEpisodeButton.addEventListener('click', () => {
            // Emit to start the next episode
            socket.emit('start_game', { playerName: document.getElementById('player-name').value }, (response) => {
                console.log('Next episode started:', response);
            });
    
            // Hide the episode info and button for the next round
            document.getElementById('episode-info').style.display = 'none';
            nextEpisodeButton.style.display = 'none';
        });
    });
    
    socket.on('game_finished', function(data) {
        console.log("Received scores:", data.scores);  // Client-side console log for debugging
        displayScores(data.scores);
    });

    ppoButton.addEventListener('click', () => {
        // Emit 'ppo_action' event with acknowledgment
        socket.emit('ppo_action', (response) => {
            console.log('Server acknowledged ppo_action:', response);
        });
    });

    socket.on('game_update', function(data) {
        const gameImage = document.getElementById('game-image');
        gameImage.src = 'data:image/png;base64,' + data.image;
        document.getElementById('reward').innerText = data.reward; // Update reward display
        document.getElementById('score').innerText = data.score; // Update score display
        document.getElementById('last_score').innerText = data.last_score; // Update last score display
    });

    document.addEventListener('keydown', async function(event) {
        const key = event.key;
        const validKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "Space", "PageUp", "PageDown", "1", "2"];
        if (validKeys.includes(key)) {
            console.log(`Sending action: ${key}`);

            // Emit 'send_action' with acknowledgment and handle async response
            try {
                const response = await new Promise((resolve, reject) => {
                    socket.emit('send_action', key, (ack) => {
                        if (ack.status === 'success') {
                            resolve(ack);
                        } else {
                            reject('Action processing failed');
                        }
                    });
                });
                console.log('Action processed:', response);
                document.getElementById('action').innerText = key; // Update action display
            } catch (error) {
                console.error('Error processing action:', error);
            }
        }
    });
});