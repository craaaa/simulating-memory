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

    // Qualtrics fires addOnload more than once in some navigation paths.
    var _guardKey = '_recallLoaded_' + this.questionId;
    if (window[_guardKey]) return;
    window[_guardKey] = true;

    // --- config (rewritten per topic by the builder) ---
    var RECALL_PROMPT = "__PROMPT__";
    var MIN_SECONDS = 60;

    var container = qthis.getQuestionContainer();
    var textarea = container.querySelector("textarea");

    if (!textarea) {
        // Should not happen: this JS is only attached to a TE/ESTB question.
        return;
    }

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

    // Status line: character count plus the dwell-gate countdown.
    var status = document.createElement("div");
    status.style.cssText = "margin-top:8px;color:#666;display:flex;justify-content:space-between;";
    status.innerHTML =
        '<span id="recall-count">0 characters</span>' +
        '<span id="recall-gate"></span>';
    textarea.parentNode.appendChild(status);

    var countEl = document.getElementById("recall-count");
    var gateEl  = document.getElementById("recall-gate");

    textarea.addEventListener("input", function () {
        var n = textarea.value.length;
        countEl.textContent = n === 1 ? "1 character" : n + " characters";
    });

    // Minimum dwell, then the participant advances whenever they are ready.
    // There is no upper time limit.
    var remaining = MIN_SECONDS;
    gateEl.textContent = "You may continue in " + remaining + "s";

    var iv = setInterval(function () {
        remaining -= 1;
        if (remaining <= 0) {
            clearInterval(iv);
            gateEl.textContent = "";
            qthis.showNextButton();
        } else {
            gateEl.textContent = "You may continue in " + remaining + "s";
        }
    }, 1000);

    textarea.focus();
});

Qualtrics.SurveyEngine.addOnUnload(function () {
    // No-op; the interval self-clears when the gate opens.
});
