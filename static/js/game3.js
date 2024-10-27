
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
    const highlight = document.getElementById('highlight');
    const dropdown = document.getElementById('dropdown'); // Reference to the dropdown element


    // Page navigation
    function showPage(pageId) {
        console.log('showPage', pageId);
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
            page.style.display = 'none'; // Ensure all are hidden
        });
        const page = document.getElementById(pageId);
        if (page) {
            page.classList.add('active');
            page.style.display = 'flex'; // Show the active page
        } else {
            console.error(`Page with ID ${pageId} not found.`);
        }
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
            // console.log(`Sending action: ${key}`);

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
                // console.log('Action processed:', response);
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
        socket.emit('finish_game');
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
        data.actions.forEach((action, index) => {
            const li = document.createElement('li');
            li.textContent = action.action;
            li.setAttribute('data-index', index);
            actionList.appendChild(li);
        });

        positions = data.actions.map(action => ({
            x: action.x,
            y: action.y,
            width: action.width,
            height: action.height
        }));
        // const containerRect = overview-image-container.getBoundingClientRect();
        // console.log('Container dimensions:', containerRect.width, containerRect.height);
        // console.log('invalid Actions:', data.invalid_moves);
        // console.log('score:', data.score);
        console.log('positions:', positions);
        document.getElementById('invalid-actions').textContent = data.invalid_moves;
        document.getElementById('episode-score').textContent = data.score;
    
        // Show the correct page
        showPage('overview-page');
    });

    socket.on('finish_game', function(data) {
        console.log("Received scores:", data.scores);  // Client-side console log for debugging
        displayScores(data.scores);
    });

    function displayScores(scores) {
        const scoreList = document.getElementById('score-list');
        scoreList.innerHTML = ''; // Clear previous scores
        
        scores.forEach((score, index) => {
            const scoreItem = document.createElement('li');
            scoreItem.textContent = `Episode ${index + 1}: ${score}`;
            scoreList.appendChild(scoreItem);
        });
    }

    document.getElementById('actions').addEventListener('mouseover', (event) => { 
        const highlight = document.getElementById('highlight');
        if (!highlight) {
            console.error('Highlight element not found in the DOM');
        } else {
            console.log('Highlight element found:', highlight);
        }
        if (event.target.closest('li')) { // Ensure you're targeting an LI
            const liElement = event.target.closest('li');
            console.log('mouseover', liElement.dataset.index);
            
            const index = liElement.dataset.index;
            if (index !== undefined) { // Check that index exists
                const position = positions[index];
                console.log('positions:', positions);
                console.log('index:', index);
                console.log('position:', position);
                
                if (position) {
                    console.log('mouseover', position.x, position.y, position.width, position.height);
    
                    // Properly assign style using the position
                    highlight.style.left = `${position.x}px`;
                    highlight.style.top = `${position.y}px`;
                    highlight.style.width = `${position.width}px`;
                    highlight.style.height = `${position.height}px`;
                    highlight.style.display = 'block !important';
                    console.log('highlight.display:', highlight.style.display);
                } else {
                    console.error(`Position not found for index ${index}`);
                }
            } else {
                console.error('Index is undefined');
            }
        }
    });

    // document.getElementById('actions').addEventListener('mouseout', (event) => {
    //     // if (event.target.tagName === 'LI') {
    //     //     highlight.style.display = 'none';
    //     // }
    // });
    document.getElementById('actions').addEventListener('click', (event) => {
        if (event.target.tagName === 'LI') {
            const rect = event.target.getBoundingClientRect();
            
            // Set dropdown position relative to the clicked item
            dropdown.style.left = `${rect.right + window.scrollX}px`;
            dropdown.style.top = `${rect.top + window.scrollY}px`;
            dropdown.style.display = 'block';
            
            // Store the index for later reference
            dropdown.dataset.index = event.target.dataset.index;
        }
    });

    document.addEventListener('click', (event) => {
        if (!event.target.closest('.dropdown') && !event.target.closest('#actions li')) {
            dropdown.style.display = 'none';
        }
    });

    document.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', (event) => {
            const index = dropdown.dataset.index;
            const action = event.target.textContent;
            console.log(`Action "${action}" selected for index ${index}`);
            
            // Perform the action here as needed
            dropdown.style.display = 'none';
        });
    });    
});