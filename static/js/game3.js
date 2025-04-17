document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Script Loaded');
    
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = io.connect(protocol + '://' + document.domain + ':' + location.port);

    const coverStartButton = document.getElementById('cover-start-button');
    const welcomeContinueButton = document.getElementById('welcome-continue-button');
    const startButton = document.getElementById('start-button'); // Existing Start button
    const playerNameInput = document.getElementById('player-name');

    // Show the cover page initially
    function showCoverPage() {
        showPage('cover-page');
    }

    // Navigate to the welcome page and call start_game in the background
    coverStartButton.addEventListener('click', () => {
        console.log('Cover Start button clicked');
        socket.emit('start_game', { playerName: null, updateAgent: false }, (response) => {
            console.log('Server acknowledged start_game:', response);
        });
        showPage('welcome-page');
    });

    // Navigate to Phase 1 game page from the welcome page
    welcomeContinueButton.addEventListener('click', () => {
        const playerName = playerNameInput.value;
        if (!playerName) {
            alert("Please enter your name.");
            return;
        }

        console.log('Welcome Continue button clicked');
        socket.emit('start_game', { playerName: playerName, updateAgent: false }, (response) => {
            console.log('Server acknowledged start_game:', response);
        });

        showPage('ph1-game-page');
    });

    // Utility function to show a page
    function showPage(pageId) {
        console.log('showPage', pageId);
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
            page.style.display = 'none'; 
        });
        const page = document.getElementById(pageId);
        if (page) {
            page.classList.add('active');
            page.style.display = 'flex'; 

            // Reset the Start Agent button style when opening the ph2-game-page
            if (pageId === 'ph2-game-page') {
                const startAgentButton = document.getElementById('start-agent-button');
                if (startAgentButton) {
                    startAgentButton.style.backgroundColor = ''; // Reset to default
                    startAgentButton.style.color = ''; // Reset to default
                }
            }

            hideLoadingOverlay(); // Hide loading overlay after showing the page
        } else {
            console.error(`Page with ID ${pageId} not found.`);
        }
    }

    // Show the cover page on load
    showCoverPage();

    const ppoButton = document.getElementById('ppo-button');
    const agentPlayAllButton = document.getElementById('agent_play_all'); 
    const nextEpisodeButton = document.getElementById('next-episode-button');
    const anotherExampleButton = document.getElementById('another-example-button');
    const compareAgentsButton = document.getElementById('compare-agents-button');
    const finishGameButton = document.getElementById('finish-game-button');
    const gotoPhase2Button = document.getElementById('goto-phase2-button');
    gotoPhase2Button.style.display = 'none';
    const gameImagePh1 = document.getElementById('game-image-ph1');
    const gameImagePh2 = document.getElementById('game-image-ph2');
    const scoreList = document.getElementById('score-list');
    const placeholderIconPh = document.querySelector('#game-image-container .placeholder-icon');

    // Overview page elements
    const overviewGameImage = document.getElementById('overview-game-image');
    const currentActionElement = document.getElementById('current-action');
    const dropdown = document.getElementById('action-dropdown');
    const prevActionButton = document.getElementById('prev-action-button');
    const nextActionButton = document.getElementById('next-action-button');

    // let selectedAction = null;
    let phase1_counter = 1;
    let actions = [];  // Will store the (img, action) tuples from server
    let currentActionIndex = 0;
    let feedbackImages = []; // Define feedbackImages in the correct scope
    let userFeedback = []; // List to store user changes
    let simillarity_level = 0; // Initialize simillarity_level

    const actionsNameList = ['forward', 'turn right', 'turn left', 'pickup'];

    // Populate dropdown with actions
    function populateDropdown() {
        dropdown.innerHTML = ''; // Clear existing items
        actionsNameList.forEach(action => {
            const div = document.createElement('div');
            div.classList.add('dropdown-item');
            div.textContent = action;
            div.addEventListener('click', () => {
                handleActionSelection(currentActionIndex, action);
                dropdown.style.display = 'none';
            });
            dropdown.appendChild(div);
        });
    }

    // Show the loading overlay
    function showLoadingOverlay() {
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.style.display = 'flex';
        loadingOverlay.style.position = 'fixed'; // Ensure it stays on top
        loadingOverlay.style.zIndex = '9999'; // High z-index to overlay everything
    }

    // Hide the loading overlay
    function hideLoadingOverlay() {
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.style.display = 'none';
    }

    // Page navigation
    function showPage(pageId) {
        console.log('showPage', pageId);
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
            page.style.display = 'none'; 
        });
        const page = document.getElementById(pageId);
        if (page) {
            page.classList.add('active');
            page.style.display = 'flex'; 

            hideLoadingOverlay(); // Hide loading overlay after showing the page
        } else {
            console.error(`Page with ID ${pageId} not found.`);
        }
    }
    

    // // Page navigation
    // function showPage(pageId) {
    //     console.log('showPage', pageId);

    //     // Show the loading overlay only for pages other than 'ph1-game-page' and 'ph2-game-page'
    //     // if (pageId !== 'ph1-game-page' && pageId !== 'ph2-game-page') {
    //     if (pageId === 'compare-agent-update-page'){
    //         showLoadingOverlay();
    //     }

    //     // Simulate a delay (e.g., for server response or rendering)
    //     setTimeout(() => {
    //         document.querySelectorAll('.page').forEach(page => {
    //             page.classList.remove('active');
    //             page.style.display = 'none'; 
    //         });
    //         const page = document.getElementById(pageId);
    //         if (page) {
    //             page.classList.add('active');
    //             page.style.display = 'flex'; 
    //         } else {
    //             console.error(`Page with ID ${pageId} not found.`);
    //         }

    //         // Hide the loading overlay after the transition
    //         hideLoadingOverlay();
    //     }, 500); // Adjust the delay as needed
    // }

    function updateGotoPhase2ButtonVisibility() { // in case we want to force the user to play the toturial first
        if (phase1_counter > 0) { 
            gotoPhase2Button.style.display = 'block';
        } else {
            gotoPhase2Button.style.display = 'none';
        }
    }
    updateGotoPhase2ButtonVisibility();

    // Start button: Phase 1
    startButton.addEventListener('click', () => {
        const playerName = playerNameInput.value;
        if (!playerName) {
            alert("Please enter your name.");
            return;
        }

        console.log('Emitting start_game event');
        socket.emit('start_game', { playerName: playerName, updateAgent: false }, (response) => {
            console.log('Server acknowledged start_game:', response);
        });

        showPage('ph1-game-page');
    });

    if (gotoPhase2Button) {
        gotoPhase2Button.addEventListener('click', () => {
            console.log('Going to phase 2');
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false });
            // socket.emit('play_entire_episode');
            showPage('ph2-game-page');
        });
    }

    // if (agentPlayAllButton) {
    //     agentPlayAllButton.addEventListener('click', () => {
    //         socket.emit('play_entire_episode');
    //     });
    // }

    // Start Agent button for Phase 2
    const startAgentButton = document.getElementById('start-agent-button');
    if (startAgentButton) {
        startAgentButton.addEventListener('click', () => {
            // Change the button's background color to darker gray
            startAgentButton.style.backgroundColor = '#a9a9a9'; // Dark gray
            startAgentButton.style.color = '#fff'; // Optional: Change text color to white for better contrast
            console.log('Start Agent button clicked');
            
            // Emit the event to start the agent
            socket.emit('play_entire_episode');
        });
    }

    // Keydown events for Phase 1 only
    document.addEventListener('keydown', async function (event) {
        const activePage = document.querySelector('.page.active');
        if (activePage && activePage.id === 'ph1-game-page') {
            const key = event.key;
            console.log('Key pressed:', key);
            const validKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "Space", "PageUp", "PageDown", "1", "2"];
            if (validKeys.includes(key)) {  
                try {
                    console.log('valid key - ', key);
                    const response = await new Promise((resolve, reject) => {
                        socket.emit('send_action', key, (ack) => {
                            if (ack && ack.status === 'success') {
                                resolve(ack);
                            } else {
                                reject('Action processing failed');
                            }
                        });
                    });
                    document.getElementById('action').innerText = key; 
                    console.log('Action recorded: ', response.action);
                } catch (error) {
                    console.error('Error processing action:', error);
                }
            }
        }
    });

    // Move to Phase 2 game page from overview
    nextEpisodeButton.addEventListener('click', () => {
        console.log('Next Episode button clicked');
        simillarity_level = 0; // Reset simillarity_level for the next comparison
        console.log('update simillarity_level=', simillarity_level);
        socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions });
        // socket.emit('play_entire_episode');
        showPage('ph2-game-page');
    });

    const nextEpisodeButtonCompare = document.getElementById('next-episode-compare-button');
    if (nextEpisodeButtonCompare) {
        nextEpisodeButtonCompare.addEventListener('click', () => {
            console.log('Next Episode button clicked (Compare Agent Page)');
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false});
            // socket.emit('play_entire_episode');
            showPage('ph2-game-page');
        });
    }

    anotherExampleButton.addEventListener('click', () => {
        simillarity_level += 1; // Increment simillarity_level for the next comparison
        socket.emit('compare_agents', { playerName: playerNameInput.value, updateAgent: false , simillarity_level: simillarity_level});
        showLoadingOverlay();
        showPage('compare-agent-update-page');
    });
    
    compareAgentsButton.addEventListener('click', () => {
        socket.emit('compare_agents', { playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions, simillarity_level: simillarity_level });
        showLoadingOverlay();
        showPage('compare-agent-update-page');
    });

    finishGameButton.addEventListener('click', () => {
        socket.emit('finish_game');
        showPage('summary-page');
    });

    // Handle game updates
    socket.on('game_update', (data) => {
        const activePage = document.querySelector('.page.active');
        console.log('Game update:', activePage.id, data);
        if (activePage.id === 'ph1-game-page') {
            gameImagePh1.src = 'data:image/png;base64,' + data.image;
            placeholderIconPh.style.display = 'none';
            if (data.action) {
                document.getElementById('action').innerText = data.action;
            }
            console.log('ph1 game update data:', data.action, data.reward, data.score, data.last_score, data.step_count);
            document.getElementById('reward').innerText = data.reward;
            document.getElementById('score').innerText = data.score;
            document.getElementById('last_score').innerText = data.last_score;

            // Update step count for Phase 1
            const stepCountElementPh1 = document.getElementById('step-count-ph1');
            if (stepCountElementPh1) {
                stepCountElementPh1.innerText = data.step_count;
            }
        } else if (activePage.id === 'ph2-game-page') {
            console.log('ph2 game update data:', data.action, data.reward, data.score, data.last_score, data.step_count);
            gameImagePh2.src = 'data:image/png;base64,' + data.image;
            if (data.action) {
                document.getElementById('action2').innerText = data.action;
            }
            document.getElementById('reward2').innerText = data.reward;
            document.getElementById('score2').innerText = data.score;
            document.getElementById('last_score2').innerText = data.last_score;

            // Update step count for Phase 2
            const stepCountElementPh2 = document.getElementById('step-count-ph2');
            if (stepCountElementPh2) {
                stepCountElementPh2.innerText = data.step_count;
            }
        }
    });

    // Compare agents handler
    socket.on('compare_agents', function(data) {
        const prevAgentImage = document.getElementById('previous-agent-image');
        prevAgentImage.src = 'data:image/png;base64,' + data.prev_path_image;

        const updatedAgentImage = document.getElementById('updated-agent-image');
        updatedAgentImage.src = 'data:image/png;base64,' + data.path_image;

        showPage('compare-agent-update-page');
    });

    // When episode finishes
    socket.on('episode_finished', (data) => {
        console.log('Episode finished:', data);
        const activePage = document.querySelector('.page.active'); 

        if (activePage && activePage.id === 'ph1-game-page') {
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false });
            // if to show the finish tutorial button after the first episode
            // phase1_counter += 1;
            // if (phase1_counter > 1) {
            //     gotoPhase2Button.style.display = 'block';
            // }
            showPage('ph1-game-page');
        } else {
            // Phase 2 has finished, update the overview page
            updateOverviewPage(data);
        }
    });

    // Update the overview page with the actions array and current action
    function updateOverviewPage(data) {
        // Store actions
        actions = data.actions; 
        currentActionIndex = 0;

        // Set invalid actions and score
        // document.getElementById('invalid-actions').textContent = data.invalid_moves;
        // document.getElementById('episode-score').textContent = data.score;
        feedbackImages = data.feedback_images;

        showCurrentAction();

        showPage('overview-page');
    }

    function showCurrentAction() {
        if (actions.length === 0) return;
        const currentAction = actions[currentActionIndex];
        console.log('Current action index:', currentActionIndex, currentAction);
        overviewGameImage.src = 'data:image/png;base64,' + feedbackImages[currentActionIndex];
        currentActionElement.textContent = currentAction.action;
        //optionally remove the selected-action class from the previous action
        currentActionElement.classList.remove('selected-action');
    }

    // Navigation actions buttons
    prevActionButton.addEventListener('click', () => {
        if (currentActionIndex > 0) {
            currentActionIndex--;
            showCurrentAction();
        }
        else {
            console.log('No previous action');
        }
    });

    nextActionButton.addEventListener('click', () => {
        if (currentActionIndex < actions.length - 1) {
            currentActionIndex++;
            showCurrentAction();
        }
        else {
            console.log('No next action');
        }
    });

    // Show dropdown on current action click
    currentActionElement.addEventListener('click', (event) => {
        const rect = currentActionElement.getBoundingClientRect();
        dropdown.style.left = `${rect.left}px`;
        dropdown.style.top = `${rect.bottom + window.scrollY}px`;
        dropdown.dataset.index = currentActionIndex;
        dropdown.style.display = 'block';
    });

    // Hide dropdown if clicked outside
    document.addEventListener('click', (event) => {
        if (!event.target.closest('#current-action') && !event.target.closest('.dropdown')) {
            dropdown.style.display = 'none';
        }
    });

    // Each dropdown item changes the current action when clicked
    const dropdownItems = dropdown.querySelectorAll('.dropdown-item');
    dropdownItems.forEach((item) => {
        item.addEventListener('click', () => {
            const newAction = item.textContent;
            handleActionSelection(currentActionIndex, newAction);
            // Hide the dropdown
            dropdown.style.display = 'none';
        });
    });

    function handleActionSelection(index, newAction) {
        console.log('Selected action:', newAction);
        actions[index].action = newAction;
        currentActionElement.textContent = newAction;
        currentActionElement.classList.add('selected-action');

        // Save user change
        userFeedback.push({ index: index, feedback_action: newAction });

        // Send the chosen action to the server
        fetch('/update_action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                index: index,
                action: newAction
            })
        }).then(response => response.json())
          .then(data => {
            console.log('Action updated on server:', data);
          })
          .catch(error => console.error('Error updating action:', error));
    }

    // Populate the dropdown with actions
    populateDropdown();

    socket.on('finish_game', function(data) {
        console.log("Received scores:", data.scores);
        displayScores(data.scores);
    });

    function displayScores(scores) {
        scoreList.innerHTML = '';
        scores.forEach((score, index) => {
            const scoreItem = document.createElement('li');
            scoreItem.textContent = `Episode ${index + 1}: ${score}`;
            scoreList.appendChild(scoreItem);
        });
    }
});
