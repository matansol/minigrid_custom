
document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Script Loaded');
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = io.connect(protocol + '://' + document.domain + ':' + location.port);
    
    const startButton = document.getElementById('start-button');
    const ppoButton = document.getElementById('ppo-button');
    const playEpisodeButton = document.getElementById('play-episode-button'); 
    const nextEpisodeButton = document.getElementById('next-episode-button');
    const finishGameButton = document.getElementById('finish-game-button');
    const gameImage = document.getElementById('game-image');
    const playerNameInput = document.getElementById('player-name');
    const actionList = document.getElementById('actions');
    const scoreList = document.getElementById('score-list');
    
    // Page navigation
    function showPage(pageId) {
        console.log('showPage', pageId);
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
            page.style.display = 'none'; // Ensure all are hidden
        });
        const page = document.getElementById(pageId);
        if (page) {
            console.log('showPage', pageId, 'found');
            page.classList.add('active');
            page.style.display = 'flex'; // Show the active page
        } else {
            console.error(`Page with ID ${pageId} not found.`);
        }
        console.log('showPage end', pageId);
    }

    // Welcome page: start the game
    startButton.addEventListener('click', () => {
        const playerName = playerNameInput.value;
        if (!playerName) {
            alert("Please enter your name.");
            return;
        }

        console.log('Emitting start_game event');
        socket.emit('start_game', { playerName: playerName }, (response) => {
            console.log('Server acknowledged start_game:', response);
        });

        // Transition to game page
        showPage('game-page');
    });



    // Game page: handle agent actions
    ppoButton.addEventListener('click', () => {
        socket.emit('ppo_action');
    });

    playEpisodeButton.addEventListener('click', () => {
        socket.emit('play_entire_episode');
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
                        if (ack && ack.status === 'success') {  // Safely check if ack is defined and has 'status'
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

    // Overview page: show actions and next episode button
    nextEpisodeButton.addEventListener('click', () => {
        socket.emit('start_game', { playerName: playerNameInput.value });
        showPage('game-page');
    });

    finishGameButton.addEventListener('click', () => {
        showPage('summary-page');
    });

    // When the game updates (new action, score, etc.)
    socket.on('game_update', (data) => {
        gameImage.src = 'data:image/png;base64,' + data.image;
        document.getElementById('reward').innerText = data.reward;
        document.getElementById('score').innerText = data.score;
        document.getElementById('last_score').innerText = data.last_score;
        
        // If the game is finished, move to the overview page
        if (data.done) {
            showPage('overview-page');
            actionList.innerHTML = '';
            data.actions.forEach(action => {
                const li = document.createElement('li');
                li.textContent = action;
                actionList.appendChild(li);
            });
            document.getElementById('overview-game-image').src = 'data:image/png;base64,' + data.image;
        }
    });

    socket.on('episode_finished', (data) => {
        console.log('Episode finished:', data);
    
        // Update the image of the path taken
        const overviewGameImage = document.getElementById('overview-game-image');
        if (overviewGameImage) {
            overviewGameImage.src = 'data:image/png;base64,' + data.path_image;
        } else {
            console.error('Overview game image element not found.');
        }
    
        // Update the list of actions
        const actionList = document.getElementById('actions');
        actionList.innerHTML = ''; // Clear previous actions
        data.actions.forEach((action) => {
            const li = document.createElement('li');
            li.textContent = action;
            actionList.appendChild(li);
        });
    
        // Show the correct page
        showPage('overview-page');
    });

    // Summary page: display scores
    socket.on('game_finished', (data) => {
        scoreList.innerHTML = '';
        data.scores.forEach((score, index) => {
            const li = document.createElement('li');
            li.textContent = `Episode ${index + 1}: ${score}`;
            scoreList.appendChild(li);
        });
    });
});