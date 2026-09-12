// Qualtrics question JS for the free-recall question.
//
// Question type: Text Entry, Essay Text Box (TE / ESTB), Force Response OFF.
// The textarea belongs to Qualtrics so the response exports natively as the
// column <topic>_recall. This JS only enlarges it, blocks paste, and gates the
// Next button behind a minimum dwell time.
//
// Force Response is deliberately OFF: requiring text would pressure participants
// into confabulating when they remember nothing. An empty recall is a real datum
// and is excluded (or not) at analysis time.
//
// Prerequisites in the Qualtrics survey:
//   - RECALL_PROMPT below is rewritten per topic by build_recall_v1.py
//
// NOTE: no libraries, matching the <topic>_trial audio question in this survey.

Qualtrics.SurveyEngine.addOnload(function () {
    var qthis = this;

    // --- config (rewritten per topic by the builder) ---
    var RECALL_PROMPT = "__PROMPT__";
    var MIN_SECONDS = 60;

    var container = qthis.getQuestionContainer();
    var textarea = container.querySelector("textarea");

    if (!textarea) {
        // Should not happen: this JS is only attached to a TE/ESTB question.
        return;
    }

    // Qualtrics fires addOnload more than once in some navigation paths. Do NOT
    // bail out early on a repeat fire: Qualtrics re-renders the page with the
    // Next button in its default (visible) state, so returning before
    // hideNextButton() would silently drop the dwell gate. Instead remember when
    // the question was first mounted and resume the gate from there, so the
    // 60 s floor is measured once and cannot be reset or skipped by re-entry.
    var startKey = "_recallStart_" + this.questionId;
    if (!window[startKey]) {
        window[startKey] = new Date().getTime();
    }
    var alreadyElapsed = Math.floor((new Date().getTime() - window[startKey]) / 1000);

    qthis.hideNextButton();

    // Replace the question stem with the per-topic prompt. The prompt names only
    // the topic, never any passage content, so it cannot cue specific facts.
    var stem = container.querySelector(".QuestionText");
    if (stem) {
        stem.innerHTML = RECALL_PROMPT;
    }

    textarea.setAttribute("spellcheck", "false");
    textarea.style.width = "100%";
    textarea.style.minHeight = "260px";
    textarea.style.fontSize = "1em";
    textarea.style.lineHeight = "1.5";

    // Pasting would corrupt a recall measure.
    textarea.addEventListener("paste", function (e) { e.preventDefault(); });
    textarea.addEventListener("drop", function (e) { e.preventDefault(); });

    // Status line: character count plus the dwell-gate countdown. Keyed off the
    // DOM rather than a flag, so a re-render gets a fresh line and a surviving
    // one is not duplicated.
    var status = container.querySelector(".recall-status");
    if (!status) {
        status = document.createElement("div");
        status.className = "recall-status";
        status.style.cssText =
            "margin-top:8px;color:#666;display:flex;justify-content:space-between;";
        status.innerHTML =
            '<span class="recall-count">0 characters</span>' +
            '<span class="recall-gate"></span>';
        textarea.parentNode.appendChild(status);
    }

    var countEl = status.querySelector(".recall-count");
    var gateEl  = status.querySelector(".recall-gate");

    function renderCount() {
        var n = textarea.value.length;
        countEl.textContent = n === 1 ? "1 character" : n + " characters";
    }
    textarea.addEventListener("input", renderCount);
    renderCount();

    // Minimum dwell, then the participant advances whenever they are ready.
    // There is no upper time limit. `remaining` starts from the original mount
    // time, so a re-fire resumes the countdown instead of restarting it.
    var remaining = MIN_SECONDS - alreadyElapsed;

    function openGate() {
        gateEl.textContent = "";
        qthis.showNextButton();
    }

    if (remaining <= 0) {
        openGate();
    } else {
        gateEl.textContent = "You may continue in " + remaining + "s";
        var iv = setInterval(function () {
            remaining -= 1;
            if (remaining <= 0) {
                clearInterval(iv);
                openGate();
            } else {
                gateEl.textContent = "You may continue in " + remaining + "s";
            }
        }, 1000);
    }

    textarea.focus();
});

Qualtrics.SurveyEngine.addOnUnload(function () {
    // No-op; the interval self-clears when the gate opens.
});
