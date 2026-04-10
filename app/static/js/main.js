$(document).ready(function () {
    console.log("JS loaded");

    var pollInterval = null;

    function setButtonLoading(isLoading, text) {
        $('#btn-predict').prop('disabled', isLoading).text(text);
    }

    function startReadyPolling() {
        setButtonLoading(true, 'Models loading...');
        $('#model-hint').text('⏳ Server is loading models, this takes ~2 min on first start.');

        pollInterval = setInterval(function () {
            $.getJSON('/ready', function (data) {
                if (data.all_ready) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    setButtonLoading(false, 'Predict');
                    $('#model-hint').text('✅ All models ready!');
                    setTimeout(function () { $('#model-hint').text(''); }, 3000);
                } else {
                    var parts = [];
                    $.each(data.models, function (key, status) {
                        if (status !== 'ready') {
                            parts.push(key.toUpperCase() + ': ' + status);
                        }
                    });
                    $('#model-hint').text('⏳ Still loading: ' + parts.join(', '));
                }
            });
        }, 3000);
    }

    startReadyPolling();

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
        setButtonLoading(true, 'Predicting...');

        $.ajax({
            url:         '/predict',
            type:        'POST',
            data:        form_data,
            contentType: false,
            processData: false,
            timeout:     30000,

            success: function (data) {
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
                    msg = 'Request timed out. Please try again.';
                } else {
                    try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
                }
                $('#result-error').text("Error: " + msg).show();
            },

            complete: function () {
                setButtonLoading(false, 'Predict');
            }
        });
    });
});