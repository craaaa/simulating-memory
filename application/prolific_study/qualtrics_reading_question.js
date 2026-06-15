// Qualtrics question JS for the timed, non-selectable reading trial.
// Question type: Text/Graphic (Descriptive Text). Paste this into the question's JavaScript pane.
//
// Prerequisites in the Qualtrics survey:
//   - Look-and-Feel > Header: <script src="https://unpkg.com/jspsych@7.3.4"></script>
//                              <script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@1.1.3"></script>
//     NOTE: do NOT load jspsych.css — it overrides Qualtrics fonts/colours.
//     The passage uses `font: inherit; color: inherit` to follow the survey theme.
//   - Survey Flow has an Embedded Data block declaring: assigned_doc, reading_rt, reading_completed
//   - Replace STIMULI below with output from build_stimuli.py (var STIMULI = {...};)

Qualtrics.SurveyEngine.addOnload(function () {
    var qthis = this;

    // --- inlined stimuli (paste over with build_stimuli.py output) ---
    var STIMULI = {
        "birds": {
            "title": "<<PLACEHOLDER>>",
            "text": "<<PLACEHOLDER>>",
            "timer_seconds": 120
        }
    };

    var docId = "${e://Field/assigned_doc}";

    var doc = STIMULI[docId];
    if (!doc || !doc.text) {
        qthis.getQuestionContainer().innerHTML =
            "<p>Configuration error: missing stimulus for doc=" + docId + ". Please return this study on Prolific.</p>";
        return;
    }

    var passage = doc.text;
    var seconds = parseInt(doc.timer_seconds, 10);

    // Hide Qualtrics Next button while reading.
    qthis.hideNextButton();

    // Mount jsPsych container.
    var container = qthis.getQuestionContainer();
    container.innerHTML = '<div id="jspsych-target"></div>';

    // Block copy / right-click / common shortcuts at document level for this trial.
    var blockKeys = function (e) {
        var k = e.key ? e.key.toLowerCase() : "";
        if ((e.ctrlKey || e.metaKey) && ["c", "a", "x", "p", "s", "u"].indexOf(k) !== -1) {
            e.preventDefault();
        }
        if (e.key === "F12" || e.key === "PrintScreen") {
            e.preventDefault();
        }
    };
    var blockCopy = function (e) { e.preventDefault(); };
    var blockContext = function (e) { e.preventDefault(); };

    document.addEventListener("keydown", blockKeys, true);
    document.addEventListener("copy", blockCopy, true);
    document.addEventListener("contextmenu", blockContext, true);
    document.addEventListener("dragstart", blockContext, true);

    var jsPsych = initJsPsych({
        display_element: "jspsych-target",
        on_finish: function () {
            // Restore handlers.
            document.removeEventListener("keydown", blockKeys, true);
            document.removeEventListener("copy", blockCopy, true);
            document.removeEventListener("contextmenu", blockContext, true);
            document.removeEventListener("dragstart", blockContext, true);

            var rt = jsPsych.data.get().values()[0].rt;
            Qualtrics.SurveyEngine.setEmbeddedData("reading_rt", rt);
            Qualtrics.SurveyEngine.setEmbeddedData("reading_completed", "true");
            qthis.clickNextButton();
        }
    });

    // font/color: inherit overrides jspsych.css so the passage matches the
    // Qualtrics Look-and-Feel theme. Remove if jspsych.css is not loaded in
    // the survey header (in which case inheritance works automatically).
    var passageStyle =
        "font: inherit; color: inherit; " +
        "user-select: none; -webkit-user-select: none; -ms-user-select: none; " +
        "-moz-user-select: none; cursor: default; max-width: 720px; margin: 0 auto;";

    // Add spacing between paragraphs to match normal reading layout.
    var passagePStyle = "margin-top: 0; margin-bottom: 1em;";

    var html =
        '<div id="reading-timer" style="text-align:center; font-weight:bold; margin-bottom: 12px;"></div>' +
        '<div style="' + passageStyle + '" ' +
        'oncopy="return false" oncontextmenu="return false" ondragstart="return false" onselectstart="return false">' +
        passage.split("\n").map(function (p) { return "<p style='" + passagePStyle + "'>" + p + "</p>"; }).join("") +
        '</div>';

    var timeline = [{
        type: jsPsychHtmlKeyboardResponse,
        stimulus: html,
        choices: "NO_KEYS",
        trial_duration: seconds * 1000,
        on_load: function () {
            var remaining = seconds;
            var el = document.getElementById("reading-timer");
            el.textContent = "Time remaining: " + remaining + "s";
            var iv = setInterval(function () {
                remaining -= 1;
                if (remaining <= 0) {
                    clearInterval(iv);
                    el.textContent = "Time's up.";
                } else {
                    el.textContent = "Time remaining: " + remaining + "s";
                }
            }, 1000);
        }
    }];

    jsPsych.run(timeline);
});

Qualtrics.SurveyEngine.addOnUnload(function () {
    // No-op; cleanup handled in on_finish.
});
