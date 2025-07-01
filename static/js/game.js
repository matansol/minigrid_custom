document.addEventListener('DOMContentLoaded', () => {
    console.log('Script Loaded');

    // --- SOCKET.IO ---
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = io.connect(`${protocol}://${document.domain}:${location.port}`);

    // --- ELEMENTS ---
    const coverStartButton = document.getElementById('cover-start-button');
    const welcomeContinueButton = document.getElementById('welcome-continue-button');
    const ph2PlaceholderSpinner = document.getElementById('ph2-placeholder-spinner');
    const gameImagePh2 = document.getElementById('game-image-ph2');
    const overviewGameImage = document.getElementById('overview-game-image');
    const compareExplanationInput = document.getElementById('compare-explanation-input');
    const feedbackExplanationInput = document.getElementById('feedback-explanation');
    const startAgentButton = document.getElementById('start-agent-button');
    const nextEpisodeButton = document.getElementById('next-episode-button');
    const nextEpisodeSimpleButton = document.getElementById('next-episode-simple-button');
    const nextEpisodeButtonCompare = document.getElementById('next-episode-compare-button');
    const updateAgentsButton = document.getElementById('update-agent-button');
    const useOldAgentButton = document.getElementById('use-old-agent-button');
    const scoreList = document.getElementById('score-list');
    const currentActionElement = document.getElementById('current-action');
    const dropdown = document.getElementById('action-dropdown');
    const prevActionButton = document.getElementById('prev-action-button');
    const nextActionButton = document.getElementById('next-action-button');
    const loaderOverlay = document.getElementById('loader-overlay');

    // --- STATE ---
    let group = (() => {
        const qsGroup = new URLSearchParams(location.search).get('group');
        // Handle template variable case and invalid values
        if (qsGroup !== null && !qsGroup.includes('${')) {
            localStorage.setItem('qual_group', qsGroup); 
            return qsGroup;
        }
        const storedGroup = localStorage.getItem('qual_group');
        return storedGroup || '1'; // Default to '1' if no valid group is found
    })();
    console.log('Participant group =', group);

    let ph2ImageLoaded = false;
    let phase2_counter = 1;
    let actions = [];
    let currentActionIndex = 0;
    let feedbackImages = [];
    let userFeedback = [];
    let changedIndexes = [];
    let feedbackActionMap = {};
    const actionsNameList = ['forward', 'turn right', 'turn left', 'pickup'];

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
    // Set the value and disable/hide the input
    const prolificID = getProlificIdOrRandom();
    console.log("prolific ID=", prolificID)

    // --- PAGE HELPERS ---
    function showPage(pageId) {
    console.log('showPage', pageId);

    if (pageId === 'ph2-game-page') {
        if (phase2_counter > 5) { // game length - number of rounds
            showPage('summary-page');
            return;
        }
        ph2ImageLoaded = false;
        if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'block';
        if (gameImagePh2) gameImagePh2.style.visibility = 'hidden';

        // Reset startAgentButton style to initial look
        if (startAgentButton) {
            startAgentButton.style.backgroundColor = '';
            startAgentButton.style.color = '';
        }

        // Update round number in header
        const roundNumberElem = document.getElementById('round-number');
        if (roundNumberElem) {
            roundNumberElem.textContent = phase2_counter;
        }
    }

    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('active');
        p.style.display = 'none';
    });

    const page = document.getElementById(pageId);
    if (!page) return console.error(`Page "${pageId}" not found.`);
    page.classList.add('active');
    page.style.display = 'flex';

    // Update round number in overview page headline
    if (pageId === 'overview-page') {
        const overviewRoundElem = document.getElementById('overview-round-number');
        if (overviewRoundElem) {
            overviewRoundElem.textContent = phase2_counter - 1;
        }
    }
}
    function showLoader() { if (loaderOverlay) loaderOverlay.style.display = 'flex'; }
    function hideLoader() { if (loaderOverlay) loaderOverlay.style.display = 'none'; }

    // --- INITIAL PAGE ---
    showPage('welcome-page');

    // --- NAVIGATION ---
    // Remove coverStartButton navigation since we start directly on welcome-page

    welcomeContinueButton.addEventListener('click', () => {
        // No need to check for empty, as it is always set
        socket.emit('start_game', { playerName: prolificID, group, updateAgent: false }, ack => console.log('Server acknowledged start_game:', ack));
        showPage('ph2-game-page');
    });

    // --- PHASE 2 IMAGE HANDLING ---
    if (gameImagePh2) {
        gameImagePh2.onload = function () {
            if (!ph2ImageLoaded) {
                if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'none';
                gameImagePh2.style.visibility = 'visible';
                ph2ImageLoaded = true;
            }
        };
    }

    function setPh2GameImage(src) {
        if (!ph2ImageLoaded) {
            if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'block';
            if (gameImagePh2) gameImagePh2.style.visibility = 'hidden';
        }
        if (gameImagePh2) gameImagePh2.src = src;
    }

    // --- EPISODE NAVIGATION ---
    nextEpisodeButton.addEventListener('click', () => {
        const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
        socket.emit('start_game', { playerName: prolificID, updateAgent: false, userNoFeedback: true, userExplanation: feedbackExplanation });
        resetOverviewHighlights();
        showPage('ph2-game-page');
    });

    if (nextEpisodeSimpleButton) {
        nextEpisodeSimpleButton.addEventListener('click', () => {
            resetOverviewHighlights();
            socket.emit('start_game', { playerName: prolificID, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    if (nextEpisodeButtonCompare) {
        nextEpisodeButtonCompare.addEventListener('click', () => {
            const compareExplanation = compareExplanationInput ? compareExplanationInput.value : "";
            if (compareExplanationInput) compareExplanationInput.value = '';
            socket.emit('use_old_agent', { use_old_agent: false, demonstration_time: demonstrationTime, compareExplanationText: compareExplanation }, response => {
                console.log('update to database we use the updated agent---------------', response);
            });
            clearCanvas('updated-agent-canvas');
            clearCanvas('previous-agent-canvas');
            socket.emit('start_game', { playerName: prolificID, updateAgent: false, compareExplanationText: compareExplanation });
            showPage('ph2-game-page');
        });
    }

    if (useOldAgentButton) {
        useOldAgentButton.addEventListener('click', () => {
            const compareExplanation = compareExplanationInput ? compareExplanationInput.value : "";
            if (compareExplanationInput) compareExplanationInput.value = '';
            socket.emit('use_old_agent', { use_old_agent: true, demonstration_time: demonstrationTime, compareExplanationText: compareExplanation }, response => {
                console.log('update to database we use the old agent---------------------------', response);
            });
            clearCanvas('updated-agent-canvas');
            clearCanvas('previous-agent-canvas');
            socket.emit('start_game', { playerName: prolificID, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    function clearCanvas(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    // --- AGENT BUTTONS ---
    if (startAgentButton) {
        startAgentButton.addEventListener('click', () => {
            startAgentButton.style.backgroundColor = '#a9a9a9';
            startAgentButton.style.color = '#fff';
            socket.emit('play_entire_episode');
        });
    }

    // --- UPDATE AGENTS BUTTON ---
    updateAgentsButton.addEventListener('click', () => {
        const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
        if (feedbackExplanationInput) feedbackExplanationInput.value = '';
        if (userFeedback.length === 0) {
            alert('No feedback was given. You cannot update the agent without providing feedback.');
            return;
        }
        showLoader();
        socket.emit('compare_agents', {
            playerName: prolificID,
            updateAgent: true,
            userFeedback,
            actions,
            simillarity_level: group,
            feedbackExplanationText: feedbackExplanation
        });
        userFeedback = [];
        resetOverviewHighlights();
        setTimeout(() => {
            hideLoader();
            showPage(parseInt(group) === 0 ? 'simple-update-page' : 'compare-agent-update-page');
        }, 2000);
    });

    // --- SOCKET EVENTS ---
    socket.on('game_update', data => {
        const activePage = document.querySelector('.page.active');
        if (!activePage) return;
        if (activePage.id === 'ph2-game-page') {
            setPh2GameImage('data:image/png;base64,' + data.image);
            if (data.action) document.getElementById('action2').innerText = data.action;
            document.getElementById('reward2').innerText = data.reward;
            document.getElementById('score2').innerText = data.score;
            // document.getElementById('last_score2').innerText = data.last_score;
            const stepCountElementPh2 = document.getElementById('step-count-ph2');
            if (stepCountElementPh2) stepCountElementPh2.innerText = data.step_count;
        } else if (activePage.id === 'overview-page') {
            overviewGameImage.style.visibility = 'visible';
            overviewGameImage.src = 'data:image/png;base64,' + data.image;
        } else if (activePage.id === 'compare-agent-update-page') {
            previousAgentImage.style.visibility = 'visible';
            previousAgentImage.src = 'data:image/png;base64,' + data.prev_image;
            updatedAgentImage.style.visibility = 'visible';
            updatedAgentImage.src = 'data:image/png;base64,' + data.upd_image;
        }
    });

    socket.on('compare_agents', data => {
        let rawImageSrc = data.rawImage;
        if (rawImageSrc && !rawImageSrc.startsWith("data:image")) {
            rawImageSrc = "data:image/png;base64," + rawImageSrc;
        }
        let convergeActionLocation = data.convergeActionLocation || -1;
        let finishedCount = 0;
        function onCanvasDrawn() {
            finishedCount += 1;
            if (finishedCount === 2) {
                demonstrationTime = new Date().toISOString();
            }
        }
        drawPathOnCanvas('previous-agent-canvas', rawImageSrc, data.prevMoveSequence, {
            moveColor: 'yellow',
            turnColor: 'lightblue',
            pickupColor: '#e6007a',
            convergeActionLocation,
            scale: 2.2,
            margin: 20
        }, onCanvasDrawn);
        drawPathOnCanvas('updated-agent-canvas', rawImageSrc, data.updatedMoveSequence, {
            moveColor: 'yellow',
            turnColor: 'lightblue',
            pickupColor: '#e6007a',
            convergeActionLocation,
            scale: 2.2,
            margin: 0
        }, onCanvasDrawn);
    });

    socket.on('episode_finished', data => {
        const activePage = document.querySelector('.page.active');
        phase2_counter += 1;
        // Update round number in header
        const roundNumberElem = document.getElementById('round-number');
        if (roundNumberElem) {
            roundNumberElem.textContent = phase2_counter;
        }
        updateOverviewPage(data);
        
    });

    socket.on('finish_game', data => {
        displayScores(data.scores);
    });

    // --- OVERVIEW PAGE LOGIC ---
    function updateOverviewPage(data) {
        actions = data.actions.map(a => ({ ...a, orig_action: a.action }));
        currentActionIndex = 0;
        feedbackImages = data.feedback_images;
        actionsCells = data.actions_cells;
        showCurrentAction();
        showPage('overview-page');
    }

    function clearOverviewDot() {
        const canvas = document.getElementById('overview-overlay-canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    function showCurrentAction() {
        clearOverviewDot();
        if (actions.length === 0) return;
        const currentAction = actions[currentActionIndex];
        overviewGameImage.style.display = 'block';
        if (overviewGameImage.nextSibling && overviewGameImage.nextSibling.tagName === 'CANVAS') {
            overviewGameImage.parentNode.removeChild(overviewGameImage.nextSibling);
        }
        const parent = overviewGameImage.parentNode;
        if (parent && window.getComputedStyle(parent).position === 'static') {
            parent.style.position = 'relative';
        }
        overviewGameImage.src = 'data:image/png;base64,' + feedbackImages[currentActionIndex];
        currentActionElement.textContent = currentAction.action;
        currentActionElement.classList.remove('selected-action');
        if (changedIndexes.includes(currentActionIndex)) {
            currentActionElement.classList.add('selected-action');
        }
        nextActionButton.style.display = currentActionIndex >= actions.length - 1 ? 'none' : 'inline-block';
    }

    
    prevActionButton.addEventListener('click', () => {
        if (currentActionIndex > 0) {
            currentActionIndex--;
            showCurrentAction();
        }
    });

    nextActionButton.addEventListener('click', () => {
        if (currentActionIndex < actions.length - 1) {
            currentActionIndex++;
            showCurrentAction();
        }
    });

    currentActionElement.addEventListener('click', () => {
        const rect = currentActionElement.getBoundingClientRect();
        dropdown.style.left = `${rect.left}px`;
        dropdown.style.top = `${rect.bottom + window.scrollY}px`;
        dropdown.dataset.index = currentActionIndex;
        dropdown.style.display = 'block';
    });

    document.addEventListener('click', event => {
        if (!event.target.closest('#current-action') && !event.target.closest('.dropdown')) {
            dropdown.style.display = 'none';
        }
    });

    function relativeDirection(origAction, newAction, origDir) {
        let directionNumber = {0:"left", 1:"up", 2:"right", 3:"down", "left": 0, "up": 1, "right": 2, "down": 3}
        let newDir = directionNumber[origDir]
        if (origAction == "turn left"){
            newDir += 1
        }
        else if (origAction == "turn right"){
            newDir += -1
        }
        
        if (newAction == "turn left"){
            newDir += -1
        }
        else if (newAction == "turn right"){
            newDir += 1
        }
        newDir = (4+newDir) % 4;
        return directionNumber[newDir]
        
    }

    function drawActionSymbolOnOverviewImage(action, actionDir, origActionDir, col, row) {
        const img = overviewGameImage;
        if (!img) return;

        // Create or get the overlay canvas
        let canvas = document.getElementById('overview-overlay-canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = 'overview-overlay-canvas';
            canvas.style.position = 'absolute';
            canvas.style.left = img.offsetLeft + 'px';
            canvas.style.top = img.offsetTop + 'px';
            canvas.style.pointerEvents = 'none';
            img.parentNode.appendChild(canvas);
        }

        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;

        // 8x8 grid
        const gridRows = 8, gridCols = 8;
        const cellWidth = img.width / gridCols;
        const cellHeight = img.height / gridRows;

        // Center of the cell
        centerShift = {"right":[10, 12],
                        "left": [-10, 0],
                        "down": [5, 5],
                        "up": [-3, 0]
        }
        const shift = centerShift[origActionDir] || [0, 0];
        const centerX = col * cellWidth + cellWidth / 2 + shift[0];
        const centerY = row * cellHeight + cellHeight / 2 + shift[1];

        // Draw the symbol
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Parse action and direction
        let actionType = action;
        let direction = actionDir;
        if (action.startsWith('turn')) {
            actionType = 'turn';
        } else if (action.startsWith('pickup')) {
            actionType = 'pickup';
        } else if (action === 'forward') {
            actionType = 'forward';
        }

        function drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius, color) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(cx, cy - outerRadius);
            for (let i = 0; i < spikes * 2; i++) {
                const angle = Math.PI / spikes * i;
                const r = i % 2 === 0 ? outerRadius : innerRadius;
                ctx.lineTo(cx + Math.sin(angle) * r, cy - Math.cos(angle) * r);
            }
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.shadowColor = 'black';
            ctx.shadowBlur = 4;
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.restore();
        }

        // Direction vectors
        const dirVectors = {
            up:    { dx: 0, dy: -1 },
            down:  { dx: 0, dy: 1 },
            left:  { dx: -1, dy: 0 },
            right: { dx: 1, dy: 0 }
        };

        // Draw the correct symbol using the new drawArrow signature
        if (actionType === 'forward') {
            // Arrow line stops at base of arrow head
            const vec = dirVectors[direction] || dirVectors.up;
            const dx = vec.dx * cellWidth * 0.5;
            const dy = vec.dy * cellHeight * 0.5;
            const headSize = 15;
            // Compute the shortened line so it stops at the base of the arrow head
            const length = Math.sqrt(dx * dx + dy * dy);
            const lineLength = length - headSize;
            const angle = Math.atan2(dy, dx);
            const endX = centerX + (lineLength > 0 ? Math.cos(angle) * lineLength : 0);
            const endY = centerY + (lineLength > 0 ? Math.sin(angle) * lineLength : 0);

            // Draw the line
            if (lineLength > 0) {
                ctx.save();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(endX, endY);
                ctx.stroke();
                ctx.restore();
            }
            // Draw the arrow head at the tip
            drawArrow(ctx, endX, endY, dx - (length - lineLength) * Math.cos(angle), dy - (length - lineLength) * Math.sin(angle), headSize, 'white');
        } else if (actionType === 'turn') {
            // Draw a short line (stem) for the turn arrow
            const vec = dirVectors[direction] || dirVectors.right;
            const stemLen = Math.min(cellWidth, cellHeight) * 0.18; // 18% of cell size
            const tipX = centerX + vec.dx * stemLen;
            const tipY = centerY + vec.dy * stemLen;
            // Draw the short line
            ctx.save();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(tipX, tipY);
            ctx.stroke();
            ctx.restore();
            // Draw the arrow head at the tip of the short line
            const headSize = 18;
            drawArrow(ctx, tipX, tipY, vec.dx * stemLen, vec.dy * stemLen, headSize, 'white');
        } else if (actionType === 'pickup') {
            // Star in the pickup direction
            const vec = dirVectors[direction] || dirVectors.up;
            const starX = centerX + vec.dx * cellWidth * 0.15;
            const starY = centerY + vec.dy * cellHeight * 0.15;
            drawStar(
                ctx,
                starX,
                starY,
                5,
                Math.min(cellWidth, cellHeight) * 0.13,
                Math.min(cellWidth, cellHeight) * 0.05,
                'white'
            );
        }
    }

    function populateDropdown() {
        dropdown.innerHTML = '';
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
    populateDropdown();

    function handleActionSelection(index, newAction) {
        const originalAction = actions[index].orig_action;
        actions[index].action = newAction;
        userFeedback.push({ index, agent_action: originalAction, feedback_action: newAction });
        const idx = changedIndexes.indexOf(index);
        if (newAction !== originalAction) {
            if (idx === -1) changedIndexes.push(index);
            feedbackActionMap[index] = newAction;
        } else {
            if (idx !== -1) changedIndexes.splice(idx, 1);
            delete feedbackActionMap[index];
        }
        showCurrentAction();
        let newActionDir = relativeDirection(originalAction, newAction, actions[index].action_dir)
        drawActionSymbolOnOverviewImage(newAction, newActionDir, actions[index].action_dir, actionsCells[index][0], actionsCells[index][1]);
    }

    function resetOverviewHighlights() {
        changedIndexes = [];
        if (currentActionElement) currentActionElement.classList.remove('selected-action');
    }

    function displayScores(scores) {
        scoreList.innerHTML = '';
        scores.forEach((score, index) => {
            const scoreItem = document.createElement('li');
            scoreItem.textContent = `Episode ${index + 1}: ${score}`;
            scoreList.appendChild(scoreItem);
        });
    }
});

// --- DRAWING HELPERS ---
function drawPathOnCanvas(canvasId, imageSrc, moveSequence, config = {}, onFinish) {
    const scale = config.scale || 2;
    const margin = config.margin || 10;
    const startPoint = { x: 50, y: 50 };
    const arrowSize = 20;
    const arrowHeadSize = 12;
    const smallShift = 9;
    const allArrowSize = arrowSize + arrowHeadSize;
    const moveArrowSizes = {
        'up': { dx: 0, dy: -20, offsetX: 0, offsetY: -allArrowSize },
        'down': { dx: 0, dy: 20, offsetX: 0, offsetY: allArrowSize },
        'right': { dx: 20, dy: 0, offsetX: allArrowSize, offsetY: 0 },
        'left': { dx: -20, dy: 0, offsetX: -allArrowSize, offsetY: 0 }
    };
    const turnArrowSizes = {
        'up': { dx: 0, dy: -5 },
        'down': { dx: 0, dy: 5 },
        'right': { dx: 5, dy: 0 },
        'left': { dx: -5, dy: 0 }
    };
    const pickupDirection = {
        'up': { dx: 0, dy: -1 },
        'down': { dx: 0, dy: 1 },
        'left': { dx: -1, dy: 0 },
        'right': { dx: 1, dy: 0 }
    };

    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        canvas.width = img.width * scale + margin * 2;
        canvas.height = img.height * scale + margin * 2;
        ctx.drawImage(img, margin, margin, img.width * scale, img.height * scale);
        if (typeof moveSequence === "string") moveSequence = JSON.parse(moveSequence);
        let currentPoint = { x: startPoint.x * scale + margin, y: startPoint.y * scale + margin };
        moveSequence.forEach((item, i) => {
            const actionDir = item[0];
            const actionName = item[1]
            if (config.convergeActionLocation !== undefined && i === config.convergeActionLocation) {
                ctx.fillStyle = 'rgba(0, 60, 255, 0.4)';
                ctx.fillRect(currentPoint.x - 10, currentPoint.y - 10, 15, 15);
            }
            if (actionName == 'forward') {
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                    moveArrowSizes[actionDir].dx * scale, moveArrowSizes[actionDir].dy * scale,
                    10 * scale, 'cyan');
                const moveColor = config.moveColor || 'yellow';
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                    moveArrowSizes[actionDir].dx * scale, moveArrowSizes[actionDir].dy * scale,
                    10 * scale, moveColor);
                currentPoint.x += moveArrowSizes[actionDir].offsetX * scale;
                currentPoint.y += moveArrowSizes[actionDir].offsetY * scale;
            } else if (actionName.startsWith("turn")) {
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                    turnArrowSizes[actionDir].dx * scale, turnArrowSizes[actionDir].dy * scale,
                    7 * scale, 'cyan');
                const turnColor = config.turnColor || 'white';
                drawArrow(ctx, currentPoint.x, currentPoint.y,
                    turnArrowSizes[actionDir].dx * scale, turnArrowSizes[actionDir].dy * scale,
                    10 * scale, turnColor);
            } else if (actionName == 'pickup') {
                const pickup = pickupDirection[actionDir];
                const centerX = currentPoint.x + smallShift * scale * pickup.dx;
                const centerY = currentPoint.y + smallShift * scale * pickup.dy;
                const outerRadius = 6 * scale;
                const innerRadius = outerRadius * 0.4;
                const points = 5;
                ctx.fillStyle = config.pickupColor || 'pink';
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
        if (onFinish) onFinish();
    };
    img.src = imageSrc;
}

function drawArrow(ctx, x, y, dx, dy, headSize, color) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dx, y + dy);
    ctx.stroke();
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

// Store demonstration time globally
let demonstrationTime = null;