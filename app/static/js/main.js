$(document).ready(function () {
    console.log("JS loaded");

    var pollInterval = null;
    var pollAttempts = 0;
    var MAX_POLL_ATTEMPTS = 200; // 200 × 3s = 10 minutes max wait

    function setButtonLoading(isLoading, text) {
        $('#btn-predict').prop('disabled', isLoading).text(text);
    }

    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }

    function showRetryButton() {
        if ($('#btn-retry').length === 0) {
            $('<button id="btn-retry" class="btn btn-warning btn-sm" style="margin-left:0.8em;">Retry</button>')
                .appendTo('#model-hint-wrap')
                .on('click', function () {
                    $(this).remove();
                    pollAttempts = 0;
                    startReadyPolling();
                });
        }
    }

    function startReadyPolling() {
        setButtonLoading(true, 'Models loading...');
        $('#model-hint').text('⏳ Server is loading models, this may take ~2 min on first start.');

        pollInterval = setInterval(function () {
            pollAttempts++;

            $.ajax({
                url: '/ready',
                type: 'GET',
                dataType: 'json',
                timeout: 5000,

                success: function (data) {
                    if (data.all_ready) {
                        stopPolling();
                        setButtonLoading(false, 'Predict');
                        $('#model-hint').text('✅ All models ready!');
                        setTimeout(function () { $('#model-hint').text(''); }, 3000);
                        return;
                    }

                    // Check for any failed models
                    var failedModels = [];
                    var loadingModels = [];
                    $.each(data.models, function (key, status) {
                        if (status === 'failed') {
                            failedModels.push(key.toUpperCase());
                        } else if (status !== 'ready') {
                            loadingModels.push(key.toUpperCase() + ': ' + status);
                        }
                    });

                    if (failedModels.length > 0) {
                        stopPolling();
                        setButtonLoading(true, 'Model error');
                        $('#model-hint').html(
                            '❌ Failed to load: <strong>' + failedModels.join(', ') + '</strong>. ' +
                            'Click Retry to try again.'
                        );
                        showRetryButton();
                        return;
                    }

                    var percent = Math.min(Math.round((pollAttempts / MAX_POLL_ATTEMPTS) * 100), 95);
                    $('#model-hint').text('⏳ Still loading: ' + loadingModels.join(', ') + ' (' + percent + '%)');

                    if (pollAttempts >= MAX_POLL_ATTEMPTS) {
                        stopPolling();
                        setButtonLoading(true, 'Timed out');
                        $('#model-hint').text('⚠️ Models are taking too long. The server may be overloaded.');
                        showRetryButton();
                    }
                },

                error: function (xhr, textStatus) {
                    console.warn('Poll error:', textStatus, xhr.status);
                    if (pollAttempts >= MAX_POLL_ATTEMPTS) {
                        stopPolling();
                        setButtonLoading(true, 'Server unreachable');
                        $('#model-hint').text('⚠️ Cannot reach server. Please refresh the page.');
                        showRetryButton();
                    } else {
                        $('#model-hint').text('⏳ Waiting for server... (attempt ' + pollAttempts + ')');
                    }
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
            timeout:     60000,

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

    // Show image preview on file select
    $('#imageUpload').on('change', function () {
        var file = this.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function (e) {
            var $prev = $('#img-preview');
            if ($prev.length === 0) {
                $prev = $('<img id="img-preview" class="img-preview" alt="preview">').insertAfter('#imageUpload');
            }
            $prev.attr('src', e.target.result);
        };
        reader.readAsDataURL(file);
    });
});