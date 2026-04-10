$(document).ready(function () {
    console.log("JS loaded");

    // ── Show a hint when switching to heavy models ─────────────────────────
    $('#modelSelect').on('change', function () {
        var selected = $(this).val();
        if (selected === 'eff' || selected === 'mob') {
            $('#model-hint').text('⚠️ First prediction with this model may take 30–60s to load.');
        } else {
            $('#model-hint').text('');
        }
    });

    $(document).on('click', '#btn-predict', function (e) {
        e.preventDefault();
        console.log("Predict clicked");

        // Validate file
        var file_input = $('#imageUpload')[0];
        if (!file_input.files || file_input.files.length === 0) {
            alert("Please select an image file first.");
            return;
        }

        var selectedModel = $('#modelSelect').val();

        // Build form data
        var form_data = new FormData();
        form_data.append('file',  file_input.files[0]);
        form_data.append('model', selectedModel);

        // Show loading state
        $('#result-box').hide();
        $('#result-error').hide();
        $('#model-hint').text('');

        // FIX: Show a specific message for heavy models on first load
        var loadingText = (selectedModel === 'eff' || selectedModel === 'mob')
            ? 'Loading model (first time only)...'
            : 'Predicting...';
        $('#btn-predict').prop('disabled', true).text(loadingText);

        $.ajax({
            url:         '/predict',
            type:        'POST',
            data:        form_data,
            contentType: false,
            processData: false,

            // FIX: Set 90s timeout so the browser waits long enough for
            // first-time model downloads (~30-60s) without killing the request.
            timeout: 90000,

            success: function (data) {
                console.log(data);
                if (data.error) {
                    $('#result-error').text("Error: " + data.error).show();
                } else {
                    $('#result-text').text(data.result);
                    $('#result-meta').html(
                        '<span class="badge-model">' + data.model + '</span>' +
                        ' &nbsp; Confidence: <strong>' + data.confidence + '%</strong>' +
                        ' &nbsp; Class ID: <strong>' + data.class_id + '</strong>'
                    );
                    $('#result-box').show();
                    // Clear the hint once model has loaded successfully
                    $('#model-hint').text('');
                }
            },

            // FIX: Distinguish between a timeout and a real server error
            error: function (xhr, textStatus) {
                console.error(textStatus, xhr.responseText);
                var msg;
                if (textStatus === 'timeout') {
                    msg = 'Request timed out — the model is still loading. Please try again in a moment.';
                } else {
                    msg = "An error occurred.";
                    try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
                }
                $('#result-error').text("Error: " + msg).show();
            },

            complete: function () {
                $('#btn-predict').prop('disabled', false).text('Predict');
            }
        });
    });
});