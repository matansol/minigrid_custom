document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Script Loaded');
    
    /* ---------------- SOCKET.IO ---------------- */
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket   = io.connect(`${protocol}://${document.domain}:${location.port}`);

    /* ---------------- ELEMENTS ---------------- */
    const coverStartButton      = document.getElementById('cover-start-button');
    const welcomeContinueButton = document.getElementById('welcome-continue-button');
    const playerNameInput       = document.getElementById('player-name');

    // Phase-2
    const ph2PlaceholderSpinner = document.getElementById('ph2-placeholder-spinner');
    const gameImagePh2          = document.getElementById('game-image-ph2');
    // Overview
    const overviewGameImage     = document.getElementById('overview-game-image');
    // Compare-update page
    const compareExplanationInput = document.getElementById('compare-explanation-input');
    const feedbackExplanationInput = document.getElementById('feedback-explanation');
    let ph2ImageLoaded = false;
    let phase2_counter = 0;

    /* ───────── ①  READ & STORE “group” FROM URL ───────── */
    let group = (() => {
        const qsGroup = new URLSearchParams(location.search).get('group');
        if (qsGroup !== null) {
            localStorage.setItem('qual_group', qsGroup);
            return qsGroup;
        }
        return localStorage.getItem('qual_group');   // may be null
    })();
    console.log('Participant group =', group);

    /* ---------------- PAGE HELPERS ---------------- */
    function showPage(pageId) {
        console.log('showPage', pageId);

        if (pageId === 'ph2-game-page') {
            if (phase2_counter >= 5) {
                showPage('summary-page')
                return;
            }
            ph2ImageLoaded = false;
            if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'block';
            if (gameImagePh2)          gameImagePh2.style.visibility = 'hidden';
        }

        const startAgentButton = document.getElementById('start-agent-button');
        if (startAgentButton) {
            startAgentButton.style.backgroundColor = 'white';
            startAgentButton.style.color           = 'black';
        }

        document.querySelectorAll('.page').forEach(p => {
            p.classList.remove('active');
            p.style.display = 'none';
        });

        const page = document.getElementById(pageId);
        if (!page) return console.error(`Page "${pageId}" not found.`);
        page.classList.add('active');
        page.style.display = 'flex';
    }

    showPage('cover-page');          // initial

    /* ---------------- NAVIGATION ---------------- */
    coverStartButton.addEventListener('click', () => showPage('welcome-page'));

    /* ───────── ②  SINGLE HANDLER FOR “Continue” ───────── */
    welcomeContinueButton.addEventListener('click', () => {
        const playerName = playerNameInput.value.trim();
        if (!playerName) {
            return alert('Please enter your name.');
        }

        console.log('Welcome Continue button clicked');

        /* ──────── ③  SEND group WITH start_game ──────── */
        socket.emit('start_game', { playerName: playerName, group: group, updateAgent: false },
            (ack) => console.log('Server acknowledged start_game:', ack)
        );

        showPage('ph2-game-page');
    });

//     function clearCompareAgentCanvases() {
//     console.log('Clearing compare agent canvases');
//     const updatedAgentCanvas = document.getElementById('updated-agent-canvas');
//     if (updatedAgentCanvas) {
//         console.log('Clearing updated agent canvas');
//         const ctx = updatedAgentCanvas.getContext('2d');
//         ctx.clearRect(0, 0, updatedAgentCanvas.width, updatedAgentCanvas.height);
//     }
//     const previousAgentCanvas = document.getElementById('previous-agent-canvas');
//     if (previousAgentCanvas) {
//         console.log('Clearing previous agent canvas');
//         const ctx = previousAgentCanvas.getContext('2d');
//         ctx.clearRect(0, 0, previousAgentCanvas.width, previousAgentCanvas.height);
//     }
// }
    

    // Show the cover page initially
    function showCoverPage() {
        showPage('cover-page');
    }

    // Show the cover page on load
    showCoverPage();

    const ppoButton = document.getElementById('ppo-button');
    const agentPlayAllButton = document.getElementById('agent_play_all'); 
    const nextEpisodeButton = document.getElementById('next-episode-button');
    // const anotherExampleButton = document.getElementById('another-example-button');
    const updateAgentsButton = document.getElementById('update-agent-button');
    const finishGameButton = document.getElementById('finish-game-button');
    // const gotoPhase2Button = document.getElementById('goto-phase2-button');
    // gotoPhase2Button.style.display = 'none';
    const scoreList = document.getElementById('score-list');
    const placeholderIconPh = document.querySelector('#game-image-container .placeholder-icon');

    // Overview page elements
    const currentActionElement = document.getElementById('current-action');
    const dropdown = document.getElementById('action-dropdown');
    const prevActionButton = document.getElementById('prev-action-button');
    const nextActionButton = document.getElementById('next-action-button');

    // let selectedAction = null;
    let phase1_counter = 1;
    let actions =  [];  // Will store the (img, action) tuples from server
    let currentActionIndex = 0;
    let feedbackImages = []; // Define feedbackImages in the correct scope
    let userFeedback = []; // List to store user changes

    const actionsNameList = ['forward', 'turn right', 'turn left', 'pickup'];

    // // toturial is not needed anymore
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
    
    // // Keydown events for Phase 1 only
    // document.addEventListener('keydown', async function (event) {
    //     const activePage = document.querySelector('.page.active');
    //     if (activePage && activePage.id === 'ph1-game-page') {
    //         const key = event.key;
    //         console.log('Key pressed:', key);
    //         const validKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "Space", "PageUp", "PageDown", "1", "2"];
    //         if (validKeys.includes(key)) {  
    //             try {
    //                 console.log('valid key - ', key);
    //                 const response = await new Promise((resolve, reject) => {
    //                     socket.emit('send_action', key, (ack) => {
    //                         if (ack && ack.status === 'success') {
    //                             resolve(ack);
    //                         } else {
    //                             reject('Action processing failed');
    //                         }
    //                     });
    //                 });
    //                 document.getElementById('action').innerText = key; 
    //                 console.log('Action recorded: ', response.action);
    //             } catch (error) {
    //                 console.error('Error processing action:', error);
    //             }
    //         }
    //     }
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

    
   // When the image loads, hide the spinner and show the image (only first time after entering ph2)
    if (gameImagePh2) {
        gameImagePh2.onload = function() {
            if (!ph2ImageLoaded) {
                if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'none';
                gameImagePh2.style.visibility = 'visible';
                ph2ImageLoaded = true;
            }
        };
    }

    // When you set a new image, only show spinner if it's the first image after entering ph2
    function setPh2GameImage(src) {
        if (!ph2ImageLoaded) {
            if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'block';
            if (gameImagePh2) gameImagePh2.style.visibility = 'hidden';
        }
        if (gameImagePh2) {
            gameImagePh2.src = src;
        }
    }


    // Move to Phase 2 game page from overview
    nextEpisodeButton.addEventListener('click', () => {
        console.log('Next Episode button clicked');
        const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
        // socket.emit('user_feedback_explanation', { 
        //     playerName: playerNameInput.value, 
        //     explanation: feedbackExplanation, 
        //     action: 'next_round' 
        // });
        socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false, userNoFeedback: true, userExplanation: feedbackExplanation});

        resetOverviewHighlights(); // Reset highlights for overview actions
        // socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false});
        // userFeedback = []; // Clear user feedback for the next round
        // feedbackExplanationInput.value = ""; // Clear user explanation for the next round
        // socket.emit('play_entire_episode');
        showPage('ph2-game-page');
    });

    // Logic for next-episode-simple-button (same as next-episode-button)
    const nextEpisodeSimpleButton = document.getElementById('next-episode-simple-button');
    if (nextEpisodeSimpleButton) {
        nextEpisodeSimpleButton.addEventListener('click', () => {
            console.log('Next Episode (simple) button clicked');
            resetOverviewHighlights();
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    const nextEpisodeButtonCompare = document.getElementById('next-episode-compare-button');
    if (nextEpisodeButtonCompare) {
        nextEpisodeButtonCompare.addEventListener('click', () => {
            console.log('Next Episode button clicked (Compare Agent Page)');
            const compareExplanation = compareExplanationInput ? compareExplanationInput.value : "";
            if (compareExplanationInput) {
                compareExplanationInput.value = '';

                socket.emit('use_old_agent', {use_old_agent: false}, (response) => {
                    console.log('update to database we use the updated agent---------------');
                    console.log('Server response:', response);
                });
            const updatedAgentCanvas = document.getElementById('updated-agent-canvas');
            if (updatedAgentCanvas) {
                console.log('Clearing updated agent canvas');
                const ctx = updatedAgentCanvas.getContext('2d');
                ctx.clearRect(0, 0, updatedAgentCanvas.width, updatedAgentCanvas.height);
            }
            const previousAgentCanvas = document.getElementById('previous-agent-canvas');
            if (previousAgentCanvas) {
                console.log('Clearing previous agent canvas');
                const ctx = previousAgentCanvas.getContext('2d');
                ctx.clearRect(0, 0, previousAgentCanvas.width, previousAgentCanvas.height);
            }
                socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false, compareExplanationText: compareExplanation});
                // socket.emit('play_entire_episode');
                showPage('ph2-game-page');
            }
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
            });

            const updatedAgentCanvas = document.getElementById('updated-agent-canvas');
            if (updatedAgentCanvas) {
                console.log('Clearing updated agent canvas');
                const ctx = updatedAgentCanvas.getContext('2d');
                ctx.clearRect(0, 0, updatedAgentCanvas.width, updatedAgentCanvas.height);
            }
            const previousAgentCanvas = document.getElementById('previous-agent-canvas');
            if (previousAgentCanvas) {
                console.log('Clearing previous agent canvas');
                const ctx = previousAgentCanvas.getContext('2d');
                ctx.clearRect(0, 0, previousAgentCanvas.width, previousAgentCanvas.height);
            }

            // Navigate to the next episode
            socket.emit('start_game', { playerName: playerNameInput.value, updateAgent: false});
            showPage('ph2-game-page'); // Replace 'ph2-game-page' with the ID of the next episode page
        });
    }


    // Helper to reset highlights for overview actions
    function resetOverviewHighlights() {
        changedIndexes = [];
        if (currentActionElement) {
            currentActionElement.classList.remove('selected-action');
        }
    }
    
    updateAgentsButton.addEventListener('click', () => {
        const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
        if (feedbackExplanationInput) {
            feedbackExplanationInput.value = '';
        }
        console.log('Compare Agents button clicked userFeedback:', userFeedback);
        if (userFeedback.length === 0) {
            // No feedback: show message and stay on the page
            alert('No feedback was given. You cannot update the agent without providing feedback.');
            return;
        }
        showLoader(); // Show loader overlay immediately    

        // Do not call showPage here!
        socket.emit('compare_agents', { 
            playerName: playerNameInput.value, 
            updateAgent: true, 
            userFeedback: userFeedback, 
            actions: actions, 
            simillarity_level: group,
            feedbackExplanationText: feedbackExplanation
        });
        userFeedback = []; // Clear user feedback for the next round
        feedbackExplanationInput.input = ""; // Clear user explanation for the next round

        resetOverviewHighlights(); // Reset highlights for overview actions
        // After 2 seconds, hide loader and show the correct page based on group
        setTimeout(() => {
            hideLoader();
            if (parseInt(group) === 2) {
                showPage('simple-update-page');
            } else if (parseInt(group) < 2) {
                showPage('compare-agent-update-page');
            }
        }, 2000);
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
            ;
            // gameImagePh1.style.visibility = 'visible';
            // gameImagePh1.src = 'data:image/png;base64,' + data.image;
            // placeholderIconPh.style.display = 'none';
            // if (data.action) {
            //     document.getElementById('action').innerText = data.action;
            // }
            // console.log('ph1 game update data:', data.action, data.reward, data.score, data.last_score, data.step_count);
            // document.getElementById('reward').innerText = data.reward;
            // document.getElementById('score').innerText = data.score;
            // document.getElementById('last_score').innerText = data.last_score;

            // // Update step count for Phase 1
            // const stepCountElementPh1 = document.getElementById('step-count-ph1');
            // if (stepCountElementPh1) {
            //     stepCountElementPh1.innerText = data.step_count;
            // }
        } else if (activePage.id === 'ph2-game-page') {
            console.log('ph2 game update data:', data.action, data.reward, data.score, data.last_score, data.step_count);
            setPh2GameImage('data:image/png;base64,' + data.image);
            if (data.action) {
                document.getElementById('action2').innerText = data.action;
            }
            document.getElementById('reward2').innerText = data.reward;
            document.getElementById('score2').innerText = data.score;
            document.getElementById('last_score2').innerText = data.last_score;

            const stepCountElementPh2 = document.getElementById('step-count-ph2');
            if (stepCountElementPh2) {
                stepCountElementPh2.innerText = data.step_count;
            }
        } else if (activePage.id === 'overview-page') {
            overviewGameImage.style.visibility = 'visible';
            overviewGameImage.src = 'data:image/png;base64,' + data.image;
        } else if (activePage.id === 'compare-agent-update-page') {
            // You may receive separate data for each image or use same update for both
            previousAgentImage.style.visibility = 'visible';
            previousAgentImage.src = 'data:image/png;base64,' + data.prev_image;
            
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
            phase2_counter += 1;
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

    let changedIndexes = []; // Track indexes where feedback was given
    let feedbackActionMap = {}; // Map: index -> feedback action name

function handleActionSelection(index, newAction) {
    console.log('handleActionSelection called with index:', index, 'newAction:', newAction);
    const originalAction = actions[index].action;
    // Overwrite the agent action with user feedback
    actions[index].action = newAction;
    userFeedback.push({ index: index, agent_action: originalAction, feedback_action: newAction });

    // Update changedIndexes: add if not present
    const idx = changedIndexes.indexOf(index);
    if (newAction !== originalAction) {
        if (idx === -1) changedIndexes.push(index);
        feedbackActionMap[index] = newAction; // Store feedback action
    } else {
        if (idx !== -1) changedIndexes.splice(idx, 1);
        delete feedbackActionMap[index]; // Remove feedback action if reverted
    }

    showCurrentAction();
}

function showCurrentAction() {
    if (actions.length === 0) return;
    const currentAction = actions[currentActionIndex];
    overviewGameImage.style.display = 'block';
    // Remove any previous feedback canvas
    if (overviewGameImage.nextSibling && overviewGameImage.nextSibling.tagName === 'CANVAS') {
        overviewGameImage.parentNode.removeChild(overviewGameImage.nextSibling);
    }
    // Ensure parent is positioned relative for absolute overlay
    const parent = overviewGameImage.parentNode;
    if (parent && window.getComputedStyle(parent).position === 'static') {
        parent.style.position = 'relative';
    }
    overviewGameImage.onload = function() {
        // Draw feedback symbol if feedback action exists for this index
        if (feedbackActionMap[currentActionIndex]) {
            const { x, y, width, height } = actions[currentActionIndex];
            if (typeof x === 'number' && typeof y === 'number') {
                const gridSize = actions[currentActionIndex].grid_size || width || 8;
                drawFeedbackSymbolOnImageAtPos(overviewGameImage, feedbackActionMap[currentActionIndex], x, y, gridSize);
            }
        }
    };
    overviewGameImage.src = 'data:image/png;base64,' + feedbackImages[currentActionIndex];
    currentActionElement.textContent = currentAction.action;

    // Remove highlight from previous
    currentActionElement.classList.remove('selected-action');

    // Highlight if this index is in changedIndexes
    if (changedIndexes.includes(currentActionIndex)) {
        currentActionElement.classList.add('selected-action');
    }

    // Show/hide next action button
    if (currentActionIndex >= actions.length - 1) {
        nextActionButton.style.display = 'none';
    } else {
        nextActionButton.style.display = 'inline-block';
    }

    console.log('showCurrentAction feedbackActionMap:', feedbackActionMap);
    // Draw feedback symbol if feedback action exists for this index
    setTimeout(() => {
        if (feedbackActionMap[currentActionIndex]) {
            const { x, y, width, height } = actions[currentActionIndex];
            if (typeof x === 'number' && typeof y === 'number') {
                const gridSize = actions[currentActionIndex].grid_size || width || 8;
                drawFeedbackSymbolOnImageAtPos(overviewGameImage, feedbackActionMap[currentActionIndex], x, y, gridSize);
            }
        }
    }, 100); // Wait for image to load
}

function drawFeedbackSymbolOnImageAtPos(imgElement, actionName, gridX, gridY, gridSize = 8) {
    if (imgElement.width === 0 || imgElement.height === 0) {
        console.warn('Image not loaded or has zero size');
        return;
    }
    // Remove any existing feedback canvas first
    const existingCanvas = imgElement.nextSibling;
    if (existingCanvas && existingCanvas.tagName === 'CANVAS') {
        existingCanvas.remove();
    }

    const canvas = document.createElement('canvas');
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;
    canvas.style.position = 'absolute';
    canvas.style.left = imgElement.offsetLeft + 'px';
    canvas.style.top = imgElement.offsetTop + 'px';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = 10;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate sizes based on grid
    const cellW = canvas.width / gridSize;
    const cellH = canvas.height / gridSize;
    const cx = gridX * cellW + cellW / 2;
    const cy = gridY * cellH + cellH / 2;

    // Common style settings
    ctx.strokeStyle = 'orange';
    ctx.fillStyle = 'orange';
    ctx.lineWidth = 2;

    if (actionName === 'forward') {
        // Draw forward arrow (vertical line with arrowhead)
        const len = Math.min(cellW, cellH) * 0.4;
        ctx.beginPath();
        ctx.moveTo(cx, cy + len/2);
        ctx.lineTo(cx, cy - len/2);
        ctx.stroke();
        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(cx - len/4, cy - len/4);
        ctx.lineTo(cx, cy - len/2);
        ctx.lineTo(cx + len/4, cy - len/4);
        ctx.stroke();
    } else if (actionName === 'turn right' || actionName === 'turn left') {
        // Draw turn arrow (just the triangle, no stem)
        const size = Math.min(cellW, cellH) * 0.3;
        const angle = actionName === 'turn right' ? Math.PI/2 : -Math.PI/2;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(0, -size/2);
        ctx.lineTo(-size/2, size/2);
        ctx.lineTo(size/2, size/2);
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
    } else if (actionName === 'pickup') {
        // Draw star
        const points = 5;
        const outerRadius = Math.min(cellW, cellH) * 0.25;
        const innerRadius = outerRadius * 0.4;
        ctx.beginPath();
        for (let i = 0; i < 2 * points; i++) {
            const radius = i % 2 === 0 ? outerRadius : innerRadius;
            const angle = (i * Math.PI) / points - Math.PI / 2;
            const x = cx + radius * Math.cos(angle);
            const y = cy + radius * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
    }

    // Add the canvas to the DOM
    imgElement.parentNode.insertBefore(canvas, imgElement.nextSibling);
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

    // Populate the dropdown with actions
    populateDropdown();

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

// Helper to show/hide loader overlay
const loaderOverlay = document.getElementById('loader-overlay');
function showLoader() {
    if (loaderOverlay) loaderOverlay.style.display = 'flex';
}
function hideLoader() {
    if (loaderOverlay) loaderOverlay.style.display = 'none';
}

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
                const innerRadius = outerRadius * 0.4;
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
//     });
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
// })
