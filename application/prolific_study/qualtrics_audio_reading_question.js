// Qualtrics question JS for the simultaneous audio + text trial.
// Question type: Text/Graphic (Descriptive Text). Paste into the question's JavaScript pane.
//
// Prerequisites in the Qualtrics survey:
//   - Survey Flow has an Embedded Data block declaring: assigned_doc, listening_rt, listening_completed
//   - STIMULI below must have both "url" (wav) and "text" per condition.
//   - wav files hosted externally with CORS headers allowing GET from https://*.qualtrics.com.
//   - Add an audio check question before this one (see qualtrics_audio_question.js comments).

Qualtrics.SurveyEngine.addOnload(function () {
    if (window._audioReadingQuestionLoaded) return;
    window._audioReadingQuestionLoaded = true;
    var qthis = this;

    // --- inlined stimuli: each entry needs "url" (wav) and "text" ---
    var STIMULI = {
        "birds": {
            "url":  "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds.wav",
            "text": "<<PLACEHOLDER>>"
        },
        "birds_distractors": {
            "url":  "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_distractors.wav",
            "text": "<<PLACEHOLDER>>"
        },
        "birds_easier": {
            "url":  "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_easier.wav",
            "text": "<<PLACEHOLDER>>"
        },
        "birds_listed": {
            "url":  "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_listed.wav",
            "text": "<<PLACEHOLDER>>"
        },
        "birds_repeat": {
            "url":  "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_repeat.wav",
            "text": "<<PLACEHOLDER>>"
        }
    };

    var docId = "${e://Field/assigned_doc}";
    var doc = STIMULI[docId];

    if (!doc || !doc.url || !doc.text) {
        qthis.getQuestionContainer().innerHTML =
            "<p>Configuration error: missing stimulus for condition=" + docId +
            ". Please return this study on Prolific.</p>";
        return;
    }

    qthis.hideNextButton();

    var container = qthis.getQuestionContainer();
    container.innerHTML =
        '<div id="ar-status" style="text-align:center;font-weight:bold;margin-bottom:12px;">Loading audio…</div>' +
        '<div id="ar-start-wrap" style="text-align:center;margin-bottom:16px;"></div>' +
        '<div id="ar-passage" style="display:none;font:inherit;font-size:1rem;color:inherit;max-width:720px;margin:0 auto;' +
            'user-select:none;-webkit-user-select:none;-ms-user-select:none;-moz-user-select:none;' +
            'cursor:default;" oncopy="return false" oncontextmenu="return false" ' +
            'ondragstart="return false" onselectstart="return false"></div>';

    var statusEl  = document.getElementById("ar-status");
    var startWrap = document.getElementById("ar-start-wrap");
    var passageEl = document.getElementById("ar-passage");

    // Populate passage HTML (hidden until Start clicked).
    var pStyle = "margin-top:0;margin-bottom:1em;";
    passageEl.innerHTML = doc.text.split("\n")
        .map(function (p) { return "<p style='" + pStyle + "'>" + p + "</p>"; })
        .join("");

    var audio = new Audio(doc.url);
    audio.preload = "auto";
    audio.loop    = false;
    audio.addEventListener("contextmenu", function (e) { e.preventDefault(); });

    var startTime  = null;
    var progressIv = null;

    function beginTrial() {
        startTime = Date.now();
        startWrap.innerHTML = "";
        statusEl.textContent = "Audio is playing. Please read and listen.";
        passageEl.style.display = "block";

        progressIv = setInterval(function () {
            if (!isNaN(audio.duration) && audio.duration > 0) {
                var remaining = Math.ceil(audio.duration - audio.currentTime);
                if (remaining > 0) {
                    statusEl.textContent = "Audio is playing. Please read and listen. (" + remaining + "s remaining)";
                }
            }
        }, 1000);

        audio.play();
    }

    audio.addEventListener("canplaythrough", function () {
        statusEl.textContent = "Ready.";
        startWrap.innerHTML =
            '<button id="ar-start-btn" style="font-size:1em;padding:8px 24px;">▶ Start</button>';
        document.getElementById("ar-start-btn").addEventListener("click", function () {
            document.getElementById("ar-start-btn").disabled = true;
            beginTrial();
        });
    }, { once: true });

    audio.addEventListener("ended", function () {
        if (progressIv) { clearInterval(progressIv); }
        var rt = startTime ? (Date.now() - startTime) : null;
        Qualtrics.SurveyEngine.setEmbeddedData("listening_rt", rt);
        Qualtrics.SurveyEngine.setEmbeddedData("listening_completed", "true");
        statusEl.textContent = "Audio complete. Continuing…";
        setTimeout(function () { qthis.clickNextButton(); }, 2000);
    });

    audio.addEventListener("error", function () {
        if (progressIv) { clearInterval(progressIv); }
        statusEl.textContent = "Audio failed to load. Please return this study on Prolific.";
    });
});

Qualtrics.SurveyEngine.addOnUnload(function () {
    // No-op; cleanup handled in ended/error handlers above.
});
