$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.image2-section').hide();
    $('.loader').hide();
    $('#result').hide();


    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
                //empty the result box or there won't be any GET requests for /predict part when doing mutiple predicts
                $('#resultPreview').css('background-image', 'none');
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('.image2-section').hide();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });


    $('#btn-result').click(function () {
        $('.image2-section').show();
        $('#btn-predict').hide();
        $('#btn-result').hide();
        $('#resultPreview').css('background-image', 'url(/predict)');
        $('#resultPreview').hide();
        $('#resultPreview').fadeIn(650);
    });
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text('Succeed in '+data+'s!');
                //$('.image2-section').show();
                $('#btn-result').show();

                
                        
        
                console.log('Success!');


            },
        });
    });

});
