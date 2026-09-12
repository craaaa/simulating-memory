// Qualtrics question JS for the 60-second math task (filled delay).
//
// Question type: Text/Graphic (Descriptive Text), QuestionText "Loading...".
// The JS replaces the container entirely, so the placeholder text is never seen.
//
// Purpose: a Brown-Peterson style filled delay between hearing a passage and
// freely recalling it. Self-paced arithmetic occupies working memory and blocks
// rehearsal. Accuracy here is not a dependent variable of interest; it is a
// manipulation check that the participant actually engaged.
//
// Prerequisites in the Qualtrics survey:
//   - Survey Flow declares embedded data: <topic>_math_n_attempted,
//     <topic>_math_n_correct, <topic>_math_trials
//   - MATH_TOPIC below is rewritten per topic by build_recall_v1.py
//
// NOTE: this deliberately uses no libraries, matching the <topic>_trial audio
// question in this survey. Do not add jsPsych; the survey header does not load it.

Qualtrics.SurveyEngine.addOnload(function () {
    var qthis = this;

    // --- config (rewritten per topic by the builder) ---
    var MATH_TOPIC = "__TOPIC__";
    var DURATION_SECONDS = 60;

    // Qualtrics fires addOnload more than once in some navigation paths. Do NOT
    // bail out early on a repeat fire: Qualtrics re-renders the page with the
    // Next button in its default (visible) state, so returning before
    // hideNextButton() would let a participant skip the rest of the delay.
    // Keep the clock and the trial log on `window` instead, so a re-fire resumes
    // the same 60 s window rather than restarting or abandoning it.
    var stateKey = "_mathState_" + this.questionId;
    var state = window[stateKey];
    if (!state) {
        state = window[stateKey] = {
            startedAt: new Date().getTime(),
            trials: [],
            finished: false
        };
    }

    if (state.finished) {
        // The window already closed and the results are already written.
        qthis.getQuestionContainer().innerHTML =
            '<div style="text-align:center;">Time\'s up. Click Next to continue.</div>';
        qthis.showNextButton();
        return;
    }

    qthis.hideNextButton();

    var container = qthis.getQuestionContainer();
    container.innerHTML =
        '<div style="max-width:520px;margin:0 auto;text-align:center;">' +
        '  <div id="math-timer" style="font-weight:bold;margin-bottom:4px;"></div>' +
        '  <div style="color:#666;margin-bottom:20px;">' +
        '    Solve as many as you can. Type your answer and press Enter.' +
        '  </div>' +
        '  <div id="math-problem" style="font-size:2em;margin-bottom:12px;"></div>' +
        '  <div>' +
        '    <input id="math-answer" type="text" inputmode="numeric" autocomplete="off"' +
        '           style="font-size:1.5em;width:140px;text-align:center;padding:4px;">' +
        '  </div>' +
        '  <div style="margin-top:12px;">' +
        '    <button id="math-submit" type="button" style="font-size:1em;padding:6px 20px;">Submit</button>' +
        '  </div>' +
        '  <div id="math-tick" style="height:1.5em;font-size:1.2em;margin-top:12px;"></div>' +
        '</div>';

    var timerEl   = document.getElementById("math-timer");
    var problemEl = document.getElementById("math-problem");
    var answerEl  = document.getElementById("math-answer");
    var submitEl  = document.getElementById("math-submit");
    var tickEl    = document.getElementById("math-tick");

    var trials = state.trials;
    var current = null;
    var currentShownAt = null;

    function randInt(lo, hi) {
        return lo + Math.floor(Math.random() * (hi - lo + 1));
    }

    // Operator uniform over + - x. Operand sizes are chosen so every problem is
    // mental-arithmetic tractable but not automatic: two-digit operands for
    // addition and subtraction, single-digit x two-digit for multiplication.
    // Subtraction is ordered so the answer is never negative.
    function makeProblem() {
        var ops = ["+", "-", "x"];
        var op = ops[randInt(0, ops.length - 1)];
        var a, b;

        if (op === "+") {
            a = randInt(12, 89);
            b = randInt(12, 89);
            return { op: op, a: a, b: b, answer: a + b, text: a + " + " + b };
        }
        if (op === "-") {
            a = randInt(30, 99);
            b = randInt(11, a - 1);
            return { op: op, a: a, b: b, answer: a - b, text: a + " − " + b };
        }
        a = randInt(3, 9);
        b = randInt(11, 29);
        return { op: op, a: a, b: b, answer: a * b, text: a + " × " + b };
    }

    function showProblem() {
        current = makeProblem();
        currentShownAt = new Date().getTime();
        problemEl.textContent = current.text + " =";
        answerEl.value = "";
        answerEl.focus();
    }

    function commit() {
        if (state.finished || !current) return;

        var raw = answerEl.value.trim();
        if (raw === "") return;               // nothing typed; ignore the keypress

        var given = parseInt(raw, 10);
        var isCorrect = !isNaN(given) && given === current.answer;

        trials.push({
            problem: current.text,
            op: current.op,
            a: current.a,
            b: current.b,
            given: raw,
            correct: isCorrect,
            rt_ms: new Date().getTime() - currentShownAt
        });

        // Brief feedback, but no delay: the next problem is already on screen so
        // the delay stays filled for the full 60 seconds.
        tickEl.textContent = isCorrect ? "✓" : "✗";
        tickEl.style.color = isCorrect ? "#2a7" : "#c33";

        showProblem();
    }

    answerEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            commit();
        }
    });
    submitEl.addEventListener("click", function () {
        commit();
    });

    // Block paste into the answer field, and suppress the context menu, matching
    // the passage question's handling.
    answerEl.addEventListener("paste", function (e) { e.preventDefault(); });
    container.addEventListener("contextmenu", function (e) { e.preventDefault(); });

    function finish() {
        if (state.finished) return;
        state.finished = true;

        answerEl.disabled = true;
        submitEl.disabled = true;
        problemEl.textContent = "";
        tickEl.textContent = "";
        timerEl.textContent = "Time's up.";

        var nCorrect = 0;
        for (var i = 0; i < trials.length; i++) {
            if (trials[i].correct) nCorrect += 1;
        }

        Qualtrics.SurveyEngine.setEmbeddedData(MATH_TOPIC + "_math_n_attempted", trials.length);
        Qualtrics.SurveyEngine.setEmbeddedData(MATH_TOPIC + "_math_n_correct", nCorrect);
        Qualtrics.SurveyEngine.setEmbeddedData(MATH_TOPIC + "_math_trials", JSON.stringify(trials));

        qthis.clickNextButton();
    }

    // Count down from the original mount time, so a re-fire resumes the same
    // window instead of granting a fresh 60 seconds.
    var elapsed = Math.floor((new Date().getTime() - state.startedAt) / 1000);
    var remaining = DURATION_SECONDS - elapsed;

    if (remaining <= 0) {
        showProblem();
        finish();
        return;
    }

    timerEl.textContent = "Time remaining: " + remaining + "s";

    var iv = setInterval(function () {
        remaining -= 1;
        if (remaining <= 0) {
            clearInterval(iv);
            finish();
        } else {
            timerEl.textContent = "Time remaining: " + remaining + "s";
        }
    }, 1000);

    showProblem();
});

Qualtrics.SurveyEngine.addOnUnload(function () {
    // No-op; the interval is cleared in finish().
});
