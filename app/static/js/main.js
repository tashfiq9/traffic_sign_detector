$(document).ready(function () {
    console.log("JS loaded");

    // Show hint when switching to heavy models
    $('#modelSelect').on('change', function () {
        var selected = $(this).val();
        if (selected === 'eff' || selected === 'mob') {
            $('#model-hint').text('⚠️ First prediction may take up to 2 min while the model loads.');
        } else {
            $('#model-hint').text('');
        }
    });

    $(document).on('click', '#btn-predict', function (e) {
        e.preventDefault();

        var file_input = $('#imageUpload')[0];
        if (!file_input.files || file_input.files.length === 0) {
            alert("Please select an image file first.");
            return;
        }

        var selectedModel = $('#modelSelect').val();
        var form_data = new FormData();
        form_data.append('file',  file_input.files[0]);
        form_data.append('model', selectedModel);

        $('#result-box').hide();
        $('#result-error').hide();

        var loadingText = (selectedModel === 'eff' || selectedModel === 'mob')
            ? 'Loading model — please wait...'
            : 'Predicting...';
        $('#btn-predict').prop('disabled', true).text(loadingText);

        $.ajax({
            url:         '/predict',
            type:        'POST',
            data:        form_data,
            contentType: false,
            processData: false,
            timeout:     270000,   // 4.5 min — slightly longer than server's 4 min wait

            success: function (data) {
                $('#model-hint').text('');
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
                }
            },

            error: function (xhr, textStatus) {
                var msg = "An error occurred.";
                if (textStatus === 'timeout') {
                    msg = 'Request timed out. The server may be overloaded — please try again.';
                } else {
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