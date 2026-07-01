// Qualtrics question JS for the audio listening trial.
// Question type: Text/Graphic (Descriptive Text). Paste into the question's JavaScript pane.
//
// Prerequisites in the Qualtrics survey:
//   - Survey Flow has an Embedded Data block declaring: assigned_doc, listening_rt, listening_completed
//   - Replace STIMULI below with actual wav URLs per condition.
//   - wav files must be hosted externally (e.g. S3, Cloudflare R2, GitHub Pages) with CORS headers
//     allowing GET from https://*.qualtrics.com. Without this, new Audio(url) silently fails.
//   - AUDIO CHECK (native Qualtrics — no JS needed):
//       1. Create a Text/Graphic question BEFORE this one. In the Rich Content Editor,
//          click Insert > Media and upload a short test mp3 (e.g. 3 s tone).
//          Keep the embedded player visible so participants can press play themselves.
//       2. Follow it with a Multiple Choice question:
//              "Could you hear the audio clip above?"  Yes / No
//       3. In Survey Flow, add Branch Logic after that question:
//              If answer = No → jump to a "Please fix your audio / headphones and reload" page.
//     This is the recommended approach — fully native, no JS required.

Qualtrics.SurveyEngine.addOnload(function () {
    var qthis = this;
    var _guardKey = '_audioLoaded_' + this.questionId;
    if (window[_guardKey]) return;
    window[_guardKey] = true;

    var STIMULI = {
        "birds":             { "url": "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds.wav" },
        "birds_distractors": { "url": "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_distractors.wav" },
        "birds_easier":      { "url": "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_easier.wav" },
        "birds_listed":      { "url": "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_listed.wav" },
        "birds_repeat":      { "url": "https://craaaa.github.io/simulating-memory/birds_texts/20260617_birds_repeat.wav" }
    };

    var docId = "${e://Field/assigned_doc}";
    var doc = STIMULI[docId];

    if (!doc || !doc.url) {
        qthis.hideNextButton();
        qthis.getQuestionContainer().innerHTML =
            "<p>Configuration error: missing audio for condition=" + docId +
            ". Please return this study on Prolific.</p>";
        return;
    }

    qthis.hideNextButton();

    var container = qthis.getQuestionContainer();
    container.innerHTML =
        '<div id="audio-status" style="text-align:center;font-weight:bold;margin-bottom:12px;">Loading audio…</div>' +
        '<div id="audio-start-wrap" style="text-align:center;margin-bottom:12px;"></div>' +
        '<div id="audio-progress" style="text-align:center;color:#666;margin-bottom:12px;"></div>';

    var statusEl   = document.getElementById("audio-status");
    var startWrap  = document.getElementById("audio-start-wrap");
    var progressEl = document.getElementById("audio-progress");

    var audio = new Audio(doc.url);
    audio.preload = "auto";
    audio.loop    = false;

    audio.addEventListener("contextmenu", function (e) { e.preventDefault(); });

    var progressIv = null;

    function beginPlayback() {
        startWrap.innerHTML = "";
        statusEl.textContent = "Audio is playing. Please listen carefully.";

        progressIv = setInterval(function () {
            if (!isNaN(audio.duration) && audio.duration > 0) {
                var remaining = Math.ceil(audio.duration - audio.currentTime);
                progressEl.textContent = remaining > 0 ? remaining + "s remaining" : "";
            }
        }, 1000);

        audio.play();
    }

    audio.addEventListener("canplaythrough", function () {
        statusEl.textContent = "Audio ready.";
        startWrap.innerHTML =
            '<button id="audio-start-btn" style="font-size:1em;padding:8px 24px;">▶ Start Audio</button>';
        document.getElementById("audio-start-btn").addEventListener("click", function () {
            document.getElementById("audio-start-btn").disabled = true;
            beginPlayback();
        });
    }, { once: true });

    audio.addEventListener("ended", function () {
        if (progressIv) { clearInterval(progressIv); }
        var newCount = parseInt(localStorage.getItem('passage_count') || '0', 10) + 1;
        localStorage.setItem('passage_count', newCount);
        Qualtrics.SurveyEngine.setEmbeddedData("passage_count", newCount);
        Qualtrics.SurveyEngine.setEmbeddedData("listening_completed_" + newCount, "true");
        statusEl.textContent  = "You have finished passage " + newCount + " of 4. Click Next to continue.";
        progressEl.textContent = "";
        qthis.showNextButton();
        // Audio is finished and not looped — no replay possible without page reload.
    });

    audio.addEventListener("error", function () {
        if (progressIv) { clearInterval(progressIv); }
        statusEl.textContent =
            "Audio failed to load. Please return this study on Prolific.";
    });
});

Qualtrics.SurveyEngine.addOnUnload(function () {
    // No-op; cleanup handled in ended/error handlers above.
});
