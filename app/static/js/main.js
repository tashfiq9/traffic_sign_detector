$(document).ready(function () {
    console.log("JS loaded");

    // Track which models have been confirmed ready this session
    var modelsReady = { cnn: false, eff: false, mob: false };

    // ── Show hint when switching to heavy models ───────────────────────
    $('#modelSelect').on('change', function () {
        var selected = $(this).val();
        if (selected === 'eff' || selected === 'mob') {
            $('#model-hint').text('⚠️ First prediction with this model may take 30–60s to load.');
        } else {
            $('#model-hint').text('');
        }
    });

    // ── Core predict function (called directly and on retry) ──────────
    function runPredict(retryCount) {
        retryCount = retryCount || 0;

        var file_input = $('#imageUpload')[0];
        if (!file_input.files || file_input.files.length === 0) {
            alert("Please select an image file first.");
            $('#btn-predict').prop('disabled', false).text('Predict');
            return;
        }

        var selectedModel = $('#modelSelect').val();
        var form_data = new FormData();
        form_data.append('file',  file_input.files[0]);
        form_data.append('model', selectedModel);

        $.ajax({
            url:         '/predict',
            type:        'POST',
            data:        form_data,
            contentType: false,
            processData: false,
            timeout:     90000,

            success: function (data) {
                if (data.error) {
                    $('#result-error').text("Error: " + data.error).show();
                    $('#btn-predict').prop('disabled', false).text('Predict');
                } else {
                    modelsReady[selectedModel] = true;
                    $('#model-hint').text('');
                    $('#result-text').text(data.result);
                    $('#result-meta').html(
                        '<span class="badge-model">' + data.model + '</span>' +
                        ' &nbsp; Confidence: <strong>' + data.confidence + '%</strong>' +
                        ' &nbsp; Class ID: <strong>' + data.class_id + '</strong>'
                    );
                    $('#result-box').show();
                    $('#btn-predict').prop('disabled', false).text('Predict');
                }
            },

            error: function (xhr, textStatus) {
                var responseData = {};
                try { responseData = JSON.parse(xhr.responseText); } catch(e) {}
                var isStillLoading = xhr.status === 503 &&
                    responseData.error && responseData.error.indexOf('still loading') !== -1;

                if (isStillLoading && retryCount < 24) {
                    // Auto-retry every 5 seconds, up to 24 times (= 2 minutes max wait)
                    var secondsLeft = (24 - retryCount) * 5;
                    var countdown   = 5;

                    $('#model-hint').text(
                        '⏳ Models are loading on the server, retrying in ' + countdown + 's... ' +
                        '(up to ' + secondsLeft + 's remaining)'
                    );

                    // Live countdown display
                    var countInterval = setInterval(function () {
                        countdown--;
                        if (countdown > 0) {
                            $('#model-hint').text(
                                '⏳ Models are loading on the server, retrying in ' + countdown + 's... ' +
                                '(up to ' + ((24 - retryCount) * 5 - (5 - countdown)) + 's remaining)'
                            );
                        } else {
                            clearInterval(countInterval);
                        }
                    }, 1000);

                    setTimeout(function () {
                        clearInterval(countInterval);
                        runPredict(retryCount + 1);
                    }, 5000);

                } else if (textStatus === 'timeout') {
                    $('#result-error').text(
                        'Request timed out — the model is still loading. Please try again in a moment.'
                    ).show();
                    $('#btn-predict').prop('disabled', false).text('Predict');
                } else {
                    var msg = responseData.error || "An error occurred.";
                    $('#result-error').text("Error: " + msg).show();
                    $('#btn-predict').prop('disabled', false).text('Predict');
                }
            }
        });
    }

    // ── Predict button click ───────────────────────────────────────────
    $(document).on('click', '#btn-predict', function (e) {
        e.preventDefault();

        var file_input = $('#imageUpload')[0];
        if (!file_input.files || file_input.files.length === 0) {
            alert("Please select an image file first.");
            return;
        }

        var selectedModel = $('#modelSelect').val();
        var loadingText   = (selectedModel === 'eff' || selectedModel === 'mob') && !modelsReady[selectedModel]
            ? 'Loading model (first time only)...'
            : 'Predicting...';

        $('#result-box').hide();
        $('#result-error').hide();
        $('#btn-predict').prop('disabled', true).text(loadingText);

        runPredict(0);
    });
});