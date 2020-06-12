// Initialize fileinput widget
$(document).ready(function () {
    $('#input-id').fileinput({
        theme: 'fas',
        language: 'zh',
        uploadUrl: '/upload',
        allowedFileExtensions: ['jpg', 'png', 'bmp'],
        // showRemove: false,
        showClose: false,
        maxFilesNum: 1,
        maxFileCount: 1,
        browseClass: "btn btn-outline-dark",
        // previewSettings: {
        //     image: {
        //         width: "400px",
        //         height: "400px"
        //     },
        // },
    });

    var file_path = '';
    // fileinput回调处理
    $('#input-id').on('fileuploaded', function (event, data) {
        var form = data.form,
            files = data.files,
            extra = data.extra,
            response = data.response,
            reader = data.reader;
        file_path = response['file_path'];
        // console.log('fileuploaded');
        // console.log(file_path);
    });

    $('#input-id').on('fileuploaderror', function (event, data) {
        var form = data.form,
            files = data.files,
            extra = data.extra,
            response = data.response,
            reader = data.reader;
        // console.log('fileuploaderror');
        // console.log(form, response);
        alert('上传失败，请重新上传图片！');
    });

    // 开始预测
    $('#predict_result_btn').click(function(e) {
        if (file_path.length > 0) {
            $.ajax({
                type: 'POST',
                url: '/predict',
                data: {
                    "file_path": file_path
                },
                success: function(result) {
                    // console.log(result);
                    if (result['label'] > 0) {  // True
                        $('#show_result').attr('src', 'static/images/yes.png');
                    } else {
                        $('#show_result').attr('src', 'static/images/no.png');
                    }
                },
                error: function(result) {
                    // console.log(result);
                    alert('预测失败，请重试！');
                },
            });
        } else {
            alert('请先上传图片！');
        }
    });
});