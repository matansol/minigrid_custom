document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Script Loaded');
    
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = io.connect(protocol + '://' + document.domain + ':' + location.port);

    const coverStartButton = document.getElementById('cover-start-button');
    const welcomeContinueButton = document.getElementById('welcome-continue-button');
    // const startButton = document.getElementById('start-button'); // Existing Start button
    const playerNameInput = document.getElementById('player-name');

    // Elements for Phase‑1
    const ph1Spinner = document.getElementById('ph1-spinner');
    const gameImagePh1 = document.getElementById('game-image-ph1');
    // Elements for Phase‑2
    const ph2Spinner = document.getElementById('ph2-spinner');
    const gameImagePh2 = document.getElementById('game-image-ph2');
    // Elements for Overview Page
    const overviewSpinner = document.getElementById('overview-spinner');
    const overviewGameImage = document.getElementById('overview-game-image');
    // Elements for Compare Agent Update Page
    const comparePrevSpinner = document.getElementById('compare-prev-spinner');
    const compareUpdSpinner = document.getElementById('compare-upd-spinner');
    const previousAgentImage = document.getElementById('previous-agent-image');
    const updatedAgentImage = document.getElementById('updated-agent-image');
    const compareExplanationInput = document.getElementById('compare-explanation-input');
    const feedbackExplanationInput = document.getElementById('feedback-explanation');

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

        // showPage('ph1-game-page');
        showPage('ph2-game-page');
    });

    function showPage(pageId) {
        console.log('showPage', pageId);
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
            page.style.display = 'none'; 
        });
      
        // If we're about to enter a page with dynamic images, set up spinner and hide image.
        if (pageId === 'ph1-game-page') {
            ph1Spinner.style.display = 'block';
            gameImagePh1.style.visibility = 'hidden';
        }
        if (pageId === 'ph2-game-page') {
            ph2Spinner.style.display = 'block';
            gameImagePh2.style.visibility = 'hidden';
        }
        // Reset the start-agent button style when entering ph2
        const startAgentButton = document.getElementById('start-agent-button');
        if (startAgentButton) {
            console.log('Resetting start-agent button style');
            startAgentButton.style.backgroundColor = 'white';
            startAgentButton.style.color = 'black';
        }
        // if (pageId === 'overview-page') {
        //     overviewSpinner.style.display = 'block';
        //     overviewGameImage.style.visibility = 'hidden';
        // }
        if (pageId === 'compare-agent-update-page') {
            console.log('Entering compare-agent-update-page, should show single spinner');
            // Use only one spinner (comparePrevSpinner) for both images
            if (comparePrevSpinner) {
                comparePrevSpinner.style.display = 'block';
                comparePrevSpinner.innerText = 'Updating...';
            }
            if (previousAgentImage) previousAgentImage.style.visibility = 'hidden';
            if (updatedAgentImage) updatedAgentImage.style.visibility = 'hidden';

            // Hide spinner and show images after 2 seconds
            setTimeout(() => {
                if (comparePrevSpinner) {
                    comparePrevSpinner.style.display = 'none';
                    comparePrevSpinner.innerText = '';
                }
                if (previousAgentImage) previousAgentImage.style.visibility = 'visible';
                if (updatedAgentImage) updatedAgentImage.style.visibility = 'visible';
            }, 2000);
        }
      
        const page = document.getElementById(pageId);
        if (!page) {
            return console.error(`Page "${pageId}" not found.`);
        }
        page.classList.add('active');
        page.style.display = 'flex';
    
        hideLoadingOverlay();
        // updateLegendIfNeeded();
    }
    

    // Show the cover page on load
    showCoverPage();

    const ppoButton = document.getElementById('ppo-button');
    const agentPlayAllButton = document.getElementById('agent_play_all'); 
    const nextEpisodeButton = document.getElementById('next-episode-button');
    // const anotherExampleButton = document.getElementById('another-example-button');
    const compareAgentsButton = document.getElementById('compare-agents-button');
    const finishGameButton = document.getElementById('finish-game-button');
    const gotoPhase2Button = document.getElementById('goto-phase2-button');
    gotoPhase2Button.style.display = 'none';
    const scoreList = document.getElementById('score-list');
    const placeholderIconPh = document.querySelector('#game-image-container .placeholder-icon');

    // Overview page elements
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

    // toturial is not needed anymore
    // function updateGotoPhase2ButtonVisibility() { // in case we want to force the user to play the toturial first
    //     if (phase1_counter > 0) { 
    //         gotoPhase2Button.style.display = 'block';
    //     } else {
    //         gotoPhase2Button.style.display = 'none';
    //     }
    // }
    // updateGotoPhase2ButtonVisibility();

    // // Start button: Phase 2
    // startButton.addEventListener('click', () => {
    //     const playerName = playerNameInput.value;
    //     if (!playerName) {
    //         alert("Please enter your name.");
    //         return;
    //     }

    //     console.log('Emitting start_game event');
    //     socket.emit('start_game', { playerName: playerName, updateAgent: false, setEnv: true }, (response) => {
    //         console.log('Server acknowledged start_game:', response);
    //     });

    //     showPage('ph2-game-page');
    // });

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
        const feedbackExplanation = feedbackInput ? feedbackInput.value : "";
        socket.emit('user_feedback_explanation', { 
            playerName: playerNameInput.value, 
            explanation: feedbackExplanation, 
            action: 'next_round' 
        });
        socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions });
        // socket.emit('play_entire_episode');
        showPage('ph2-game-page');
    });

    const nextEpisodeButtonCompare = document.getElementById('next-episode-compare-button');
    if (nextEpisodeButtonCompare) {
        nextEpisodeButtonCompare.addEventListener('click', () => {
            console.log('Next Episode button clicked (Compare Agent Page)');
            const compareExplanation = compareExplanationInput ? compareExplanationInput.value : "";
            if (compareExplanationInput) {
                compareExplanationInput.value = '';
            }
            socket.emit('use_old_agent', {use_old_agent: false}, (response) => {
                console.log('update to database we use the updated agent---------------');
                console.log('Server response:', response);
            });
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false, compareExplanationText: compareExplanation});
            // socket.emit('play_entire_episode');
            showPage('ph2-game-page');
        });
    }

     const useOldAgentButton = document.getElementById('use-old-agent-button');
    if (useOldAgentButton) {
        useOldAgentButton.addEventListener('click', () => {
            console.log('Use Old Agent button clicked');
            const compareExplanation = compareExplanationInput ? compareExplanationInput.value : "";
            if (compareExplanationInput) {
                compareExplanationInput.value = '';
            }
            // Emit an event to the server to update the agent to the old one
            socket.emit('use_old_agent', {use_old_agent: true, compareExplanationText: compareExplanation}, (response) => {
                console.log('Server response:', response);
                console.log('update to database we use the old agent---------------------------');


                // // Temporarily change the button's color
                // useOldAgentButton.style.backgroundColor = '#a9a9a9'; // Dark gray
                // useOldAgentButton.style.color = '#fff'; // White text

                // // Reset the button's color after 2 seconds
                // setTimeout(() => {
                //     useOldAgentButton.style.backgroundColor = ''; // Reset to default
                //     useOldAgentButton.style.color = ''; // Reset to default
                // }, 2000);
            });

            // Navigate to the next episode
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false});
            showPage('ph2-game-page'); // Replace 'ph2-game-page' with the ID of the next episode page
        });
    }


    // anotherExampleButton.addEventListener('click', () => {
    //     simillarity_level += 1; // Increment simillarity_level for the next comparison
    //     socket.emit('compare_agents', { playerName: playerNameInput.value, updateAgent: false , simillarity_level: simillarity_level});
    //     showLoadingOverlay();
    //     showPage('compare-agent-update-page');
    // });
    
    compareAgentsButton.addEventListener('click', () => {
        const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
        if (feedbackExplanationInput) {
            feedbackExplanationInput.value = '';
        }
        socket.emit('compare_agents', { 
            playerName: playerNameInput.value, 
            updateAgent: true, 
            userFeedback: userFeedback, 
            actions: actions, 
            simillarity_level: simillarity_level,
            feedbackExplanationText: feedbackExplanation
        });
        showLoadingOverlay();
        showPage('compare-agent-update-page');
        // updateLegendIfNeeded();
    });

    // finishGameButton.addEventListener('click', () => {
    //     socket.emit('finish_game');
    //     showPage('summary-page');
    // });

    // Handle game updates
    socket.on('game_update', (data) => {
        const activePage = document.querySelector('.page.active');
        console.log('Game update:', activePage.id, data);
        if (activePage.id === 'ph1-game-page') {
            ph1Spinner.style.display = 'none';
            gameImagePh1.style.visibility = 'visible';
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
            ph2Spinner.style.display = 'none';
            gameImagePh2.style.visibility = 'visible';
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
        } else if (activePage.id === 'overview-page') {
            overviewSpinner.style.display = 'none';
            overviewGameImage.style.visibility = 'visible';
            overviewGameImage.src = 'data:image/png;base64,' + data.image;
        } else if (activePage.id === 'compare-agent-update-page') {
            // You may receive separate data for each image or use same update for both
            comparePrevSpinner.style.display = 'none';
            previousAgentImage.style.visibility = 'visible';
            previousAgentImage.src = 'data:image/png;base64,' + data.prev_image;
            
            compareUpdSpinner.style.display = 'none';
            updatedAgentImage.style.visibility = 'visible';
            updatedAgentImage.src = 'data:image/png;base64,' + data.upd_image;
        }
    });

    // Compare agents handler
    socket.on('compare_agents', function(data) {
        console.log('Compare agent update:', data);

        // Ensure rawImage is a valid data URL.
        let rawImageSrc = data.rawImage;
        if (rawImageSrc && !rawImageSrc.startsWith("data:image")) {
            rawImageSrc = "data:image/png;base64," + rawImageSrc;
        }

        // Ensure convergeActionLocation is a number.
        let convergeActionLocation = data.convergeActionLocation || -1;

        // Draw on the previous-agent-canvas
        drawPathOnCanvas('previous-agent-canvas', rawImageSrc, data.prevMoveSequence, {
            moveColor: 'yellow',
            turnColor: 'lightblue',
            pickupColor: 'purple',
            convergeActionLocation: convergeActionLocation,
            scale: 1.7,
            margin: 20
        });

        // Draw on the updated-agent-canvas
        drawPathOnCanvas('updated-agent-canvas', rawImageSrc, data.updatedMoveSequence, {
            moveColor: 'yellow',
            turnColor: 'lightblue',
            pickupColor: 'purple',
            convergeActionLocation: convergeActionLocation,
            scale: 1.7,
            margin: 0
        });

        // Optionally, show the compare page if not already visible.
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
        // Store actions and update current action etc.
        actions = data.actions; 
        currentActionIndex = 0;
        feedbackImages = data.feedback_images;

        showCurrentAction();

        // Show the overview page.
        showPage('overview-page');

        // Call legend update so that the legend appears in Overview page only.
        // updateLegendIfNeeded();
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
        console.log('Selected action:', newAction, 'for index:', index);
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

function drawPathOnCanvas(canvasId, imageSrc, moveSequence, config = {}) {
    console.log('Drawing path on canvas:', canvasId, imageSrc, moveSequence, config);

    // New configuration options for scaling and margin.
    const scale = config.scale || 2;    // enlarge image (default: 2x)
    const margin = config.margin || 10; // margin around image (default: 10px)

    // Constants similar to your Python code
    const startPoint = { x: 50, y: 50 };
    const arrowSize = 20;
    const arrowHeadSize = 12;
    const smallShift = 9;
    const allArrowSize = arrowSize + arrowHeadSize;

    // Settings for drawing arrows for move actions (will also be scaled)
    const moveArrowSizes = {
        'up':    { dx: 0,  dy: -20, offsetX: 0,  offsetY: -allArrowSize },
        'down':  { dx: 0,  dy: 20,  offsetX: 0,  offsetY:  allArrowSize },
        'right': { dx: 20, dy: 0,   offsetX: allArrowSize, offsetY: 0 },
        'left':  { dx: -20,dy: 0,   offsetX: -allArrowSize, offsetY: 0 }
    };
    
    const turnArrowSizes = {
        'turn up':    { dx: 0,  dy: -5 },
        'turn down':  { dx: 0,  dy: 5 },
        'turn right': { dx: 5,  dy: 0 },
        'turn left':  { dx: -5, dy: 0 }
    };
    
    const pickupDirection = {
        'up':    { dx: 0,  dy: -1 },
        'down':  { dx: 0,  dy: 1 },
        'left':  { dx: -1, dy: 0 },
        'right': { dx: 1,  dy: 0 }
    };

    // // Marker sizes for overlay (used as is; you can scale these too if needed)
    // const markSizes = {
    //     'move_vertical': { width: 25, height: 70 },
    //     'move_horizontal': { width: 80, height: 20 },
    //     'turn': { width: 20, height: 20 },
    //     'pickup': { width: 20, height: 20 }
    // };

    // Marker offsets (we will also scale these)
    // let markX = (startPoint.x + 80) * scale + margin;
    // let markY = (startPoint.y + 40) * scale + margin;

    // Obtain canvas and its context
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error("Canvas not found:", canvasId);
        return;
    }
    const ctx = canvas.getContext('2d');

    // Load the image
    const img = new Image();
    img.onload = function() {
        // Set canvas dimensions to the scaled image dimensions plus margin on both sides
        canvas.width = img.width * scale + margin * 2;
        canvas.height = img.height * scale + margin * 2;
        // Draw the image scaled and with the margin offset
        ctx.drawImage(img, margin, margin, img.width * scale, img.height * scale);
        
        // Parse moveSequence if necessary
        if (typeof moveSequence === "string") {
            moveSequence = JSON.parse(moveSequence);
        }

        // Initialize the current drawing point based on startPoint (scaled and with margin)
        let currentPoint = { x: startPoint.x * scale + margin, y: startPoint.y * scale + margin };

        moveSequence.forEach((item, i) => {
            const actionDir = item[0];

            // Optional: Draw a convergence mark if configured
            if (config.convergeActionLocation !== undefined && i === config.convergeActionLocation) {
                console_log('Drawing convergence mark at index:', i, config.convergeActionLocation);
                ctx.fillStyle = 'rgba(0, 60, 255, 0.4)';
                ctx.fillRect(currentPoint.x - 10, currentPoint.y - 10, 15, 15);
            }
            
            if (moveArrowSizes.hasOwnProperty(actionDir)) {
                // Draw feedback arrow (scale distances)
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                          moveArrowSizes[actionDir].dx * scale, moveArrowSizes[actionDir].dy * scale,
                          10 * scale, 'cyan');
                // Draw the actual arrow
                const moveColor = config.moveColor || 'yellow';
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                          moveArrowSizes[actionDir].dx * scale, moveArrowSizes[actionDir].dy * scale,
                          10 * scale, moveColor);
                
                // Update current drawing point (scaled offsets)
                currentPoint.x += moveArrowSizes[actionDir].offsetX * scale;
                currentPoint.y += moveArrowSizes[actionDir].offsetY * scale;
                
                // // Adjust marker coordinates as in your Python logic (scaled)
                // if (actionDir === 'up') {
                //     markY -= (25 + allArrowSize) * scale;
                // } else if (actionDir === 'left') {
                //     markX -= (43 + allArrowSize) * scale;
                // }
                // if (actionDir === 'down') {
                //     markY += (25 + allArrowSize) * scale;
                // } else if (actionDir === 'right') {
                //     markX += (43 + allArrowSize) * scale;
                // }
            }
            else if (turnArrowSizes.hasOwnProperty(actionDir)) {
                // Draw turn arrow (feedback then actual) with scaling
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                          turnArrowSizes[actionDir].dx * scale, turnArrowSizes[actionDir].dy * scale,
                          7 * scale, 'cyan');
                const turnColor = config.turnColor || 'white';
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                          turnArrowSizes[actionDir].dx * scale, turnArrowSizes[actionDir].dy * scale,
                          10 * scale, turnColor);
                // Adjust marker coordinates for turn (scaled)
                // const shiftSize = 17 * scale;
                // const turnShifts = {
                //     'turn up': { x: 0, y: -shiftSize },
                //     'turn down': { x: 0, y: shiftSize },
                //     'turn right': { x: shiftSize, y: 0 },
                //     'turn left': { x: -shiftSize, y: 0 }
                // };
                // const shift = turnShifts[actionDir];
                // markX += shift.x;
                // markY += shift.y;
            }
            else if (actionDir.startsWith('pickup')) {
                // Draw a pickup marker (e.g., a star), scaling offset as well
                const parts = actionDir.split(' ');
                const direction = parts[1];
                const pickup = pickupDirection[direction];
                const centerX = currentPoint.x + smallShift * scale * pickup.dx;
                const centerY = currentPoint.y + smallShift * scale * pickup.dy;
                const outerRadius = 5 * scale;
                const innerRadius = 3 * scale;
                const points = 5;

                ctx.fillStyle = config.pickupColor || 'purple';
                ctx.beginPath();
                for (let i = 0; i < 2 * points; i++) {
                    const angle = (i * Math.PI) / points;
                    const radius = i % 2 === 0 ? outerRadius : innerRadius;
                    const x = centerX + radius * Math.cos(angle);
                    const y = centerY + radius * Math.sin(angle);
                    ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.fill();
            }
        });
    };

    // Start loading the image (ensure imageSrc is a valid data URL)
    img.src = imageSrc;
}

// Helper: Draw an arrow on the given context.
function drawArrow(ctx, x, y, dx, dy, headSize, color) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    
    // Draw the line segment of the arrow
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dx, y + dy);
    ctx.stroke();
    
    // Draw the arrowhead
    const angle = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(x + dx, y + dy);
    ctx.lineTo(x + dx - headSize * Math.cos(angle - Math.PI / 6),
               y + dy - headSize * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x + dx - headSize * Math.cos(angle + Math.PI / 6),
               y + dy - headSize * Math.sin(angle + Math.PI / 6));
    ctx.lineTo(x + dx, y + dy);
    ctx.fill();
}


// function createLegend() {
//     console.log('Creating legend');
//     // Create the legend container
//     const legendContainer = document.createElement('div');
//     legendContainer.id = 'legend-container';
//     Object.assign(legendContainer.style, {
//         position: 'fixed',   // Fixed relative to viewport.
//         top: '10px',
//         left: '10px',
//         backgroundColor: 'rgba(255,255,255,0.9)',
//         padding: '8px',
//         border: '1px solid #ccc',
//         borderRadius: '3px',
//         fontFamily: 'sans-serif',
//         fontSize: '12px',
//         color: '#333',
//         zIndex: '9999'
//     });

//     // Add a title called "Ledge"
//     const titleElement = document.createElement('div');
//     titleElement.textContent = 'Ledge';
//     titleElement.style.fontWeight = 'bold';
//     titleElement.style.marginBottom = '8px';
//     legendContainer.appendChild(titleElement);

//     // Legend item for "turn" (small lightblue arrow)
//     const turnItem = document.createElement('div');
//     turnItem.style.display = 'flex';
//     turnItem.style.alignItems = 'center';
//     turnItem.style.marginBottom = '4px';
//     const turnArrow = document.createElement('div');
//     turnArrow.style.width = '0';
//     turnArrow.style.height = '0';
//     turnArrow.style.borderTop = '8px solid transparent';
//     turnArrow.style.borderBottom = '8px solid transparent';
//     // Changed arrow color from gray to lightblue
//     turnArrow.style.borderLeft = '12px solid lightblue';
//     turnArrow.style.marginRight = '6px';
//     turnItem.appendChild(turnArrow);
//     const turnText = document.createElement('span');
//     turnText.textContent = 'turn';
//     turnItem.appendChild(turnText);
//     legendContainer.appendChild(turnItem);

//     // Legend item for "move forward" (darker yellow arrow with base line)
//     const moveItem = document.createElement('div');
//     moveItem.style.display = 'flex';
//     moveItem.style.alignItems = 'center';
//     moveItem.style.marginBottom = '4px';
//     const moveArrowContainer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
//     moveArrowContainer.setAttribute('width', '20');
//     moveArrowContainer.setAttribute('height', '20');
//     moveArrowContainer.setAttribute('viewBox', '0 0 20 20');
//     moveArrowContainer.style.marginRight = '6px';
//     const baseLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
//     baseLine.setAttribute('x1', '0');
//     baseLine.setAttribute('y1', '10');
//     baseLine.setAttribute('x2', '20');
//     baseLine.setAttribute('y2', '10');
//     baseLine.setAttribute('stroke', 'darkgoldenrod');
//     baseLine.setAttribute('stroke-width', '2');
//     moveArrowContainer.appendChild(baseLine);
//     const arrowHead = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
//     arrowHead.setAttribute('points', '15,5 20,10 15,15');
//     arrowHead.setAttribute('fill', 'darkgoldenrod');
//     moveArrowContainer.appendChild(arrowHead);
//     moveItem.appendChild(moveArrowContainer);
//     const moveText = document.createElement('span');
//     moveText.textContent = 'move forward';
//     moveItem.appendChild(moveText);
//     legendContainer.appendChild(moveItem);

//     // Legend item for "pickup" (purple star)
//     const pickupItem = document.createElement('div');
//     pickupItem.style.display = 'flex';
//     pickupItem.style.alignItems = 'center';
//     const pickupStar = document.createElement('div');
//     pickupStar.textContent = '★';
//     pickupStar.style.color = 'purple';
//     pickupStar.style.fontSize = '16px';
//     pickupStar.style.marginRight = '6px';
//     pickupItem.appendChild(pickupStar);
//     const pickupText = document.createElement('span');
//     pickupText.textContent = 'pickup';
//     pickupItem.appendChild(pickupText);
//     legendContainer.appendChild(pickupItem);

//     // Append the legend container to document.body so it always appears on the left
//     document.body.appendChild(legendContainer);
// }

// // Call createLegend only when the active page is Overview or Compare Agent Update.
// function updateLegendIfNeeded() {
//     const overviewPage = document.getElementById('overview-page');
//     const comparePage = document.getElementById('compare-agent-update-page');
//     if ((overviewPage && overviewPage.classList.contains('active')) ||
//         (comparePage && comparePage.classList.contains('active'))) {
//         // Remove any existing legend to avoid duplicates
//         const existingLegend = document.getElementById('legend-container');
//         if (existingLegend) {
//             existingLegend.remove();
//         }
//         createLegend();
//     } else {
//         // If not on one of the target pages, remove the legend if it exists.
//         const existingLegend = document.getElementById('legend-container');
//         if (existingLegend) {
//             existingLegend.remove();
//         }
//     }
// }

// // Example: Call updateLegendIfNeeded after switching to a target page.
// // Below are examples for the Overview and Compare Agent Update page event handlers.
// compareAgentsButton.addEventListener('click', () => {
//     socket.emit('compare_agents', { 
//         playerName: playerNameInput.value, 
//         updateAgent: true, 
//         userFeedback: userFeedback, 
//         actions: actions, 
//         simillarity_level: simillarity_level 
//     });
//     showLoadingOverlay();
//     showPage('compare-agent-update-page');
//     // updateLegendIfNeeded();
// });

// function updateOverviewPage(data) {
//     actions = data.actions; 
//     currentActionIndex = 0;
//     feedbackImages = data.feedback_images;
    
//     showCurrentAction();
//     showPage('overview-page');
//     // updateLegendIfNeeded();
// }

// document.addEventListener('DOMContentLoaded', () => {
//     // Existing initialization code...
//     updateLegendIfNeeded();
// });
