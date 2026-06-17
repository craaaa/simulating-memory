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

    // --- inlined stimuli: replace <<PLACEHOLDER>> with hosted wav URLs ---
    var STIMULI = {
        "birds": {
            "url": "<<PLACEHOLDER_WAV_URL>>"
        }
    };

    var docId = "${e://Field/assigned_doc}";
    var doc = STIMULI[docId];

    if (!doc || !doc.url) {
        qthis.getQuestionContainer().innerHTML =
            "<p>Configuration error: missing audio for condition=" + docId +
            ". Please return this study on Prolific.</p>";
        return;
    }

    qthis.hideNextButton();

    var container = qthis.getQuestionContainer();
    container.innerHTML =
        '<div id="audio-status" style="text-align:center;font-weight:bold;margin-bottom:12px;">Preparing audio…</div>' +
        '<div id="audio-progress" style="text-align:center;color:#666;margin-bottom:12px;"></div>';

    var statusEl  = document.getElementById("audio-status");
    var progressEl = document.getElementById("audio-progress");

    var audio = new Audio(doc.url);
    audio.preload = "auto";
    audio.loop    = false;

    // Prevent right-click on the (invisible) audio element.
    audio.addEventListener("contextmenu", function (e) { e.preventDefault(); });

    var startTime = null;
    var progressIv = null;

    audio.addEventListener("canplaythrough", function () {
        statusEl.textContent = "Audio is playing. Please listen carefully.";
        startTime = Date.now();
        audio.play().catch(function () {
            // Autoplay blocked — prompt a click.
            statusEl.innerHTML =
                '<button id="audio-start-btn" style="font-size:1em;padding:8px 20px;">Click to start audio</button>';
            document.getElementById("audio-start-btn").addEventListener("click", function () {
                document.getElementById("audio-start-btn").disabled = true;
                startTime = Date.now();
                audio.play();
                statusEl.textContent = "Audio is playing. Please listen carefully.";
            });
        });

        progressIv = setInterval(function () {
            if (!isNaN(audio.duration) && audio.duration > 0) {
                var remaining = Math.ceil(audio.duration - audio.currentTime);
                progressEl.textContent = remaining > 0 ? remaining + "s remaining" : "";
            }
        }, 1000);
    }, { once: true });

    audio.addEventListener("ended", function () {
        if (progressIv) { clearInterval(progressIv); }
        var rt = startTime ? (Date.now() - startTime) : null;
        Qualtrics.SurveyEngine.setEmbeddedData("listening_rt", rt);
        Qualtrics.SurveyEngine.setEmbeddedData("listening_completed", "true");
        statusEl.textContent  = "Audio complete. Click Next to continue.";
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
