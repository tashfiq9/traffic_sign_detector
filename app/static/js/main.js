$(document).ready(function () {
    console.log("JS loaded");

    $(document).on('click', '#btn-predict', function (e) {
        e.preventDefault();
        console.log("Predict clicked");

        // Validate file
        var file_input = $('#imageUpload')[0];
        if (!file_input.files || file_input.files.length === 0) {
            alert("Please select an image file first.");
            return;
        }

        // Build form data — include the selected model
        var form_data = new FormData();
        form_data.append('file',  file_input.files[0]);
        form_data.append('model', $('#modelSelect').val());

        // Show loading state
        $('#result-box').hide();
        $('#result-error').hide();
        $('#btn-predict').prop('disabled', true).text('Predicting...');

        $.ajax({
            url:         '/predict',
            type:        'POST',
            data:        form_data,
            contentType: false,
            processData: false,
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
                }
            },
            error: function (xhr) {
                console.error(xhr.responseText);
                var msg = "An error occurred.";
                try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
                $('#result-error').text("Error: " + msg).show();
            },
            complete: function () {
                $('#btn-predict').prop('disabled', false).text('Predict');
            }
        });
    });
});