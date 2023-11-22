Для конвертации надо сбилдить контейнер и потом запустить этот контейнер из дерриктории где находятся модели.
Файл convert_trt.py запускает benchmarc на моделях osnet.pt потом сконвертированных в TorchScript, tensorrt float32, tensorrt half, с прочитанной из файла моделью tensorrt half. Конвертация в tensorrt производилась с помощью библиотеки torch-tensor.
Файл convert_onnx_trt.py пытается сконвертировать нашу модель из предварительно сконвертированной в onnx формат, модели osnet в формат tensorrt с помощью библиотеки tensorrt. И вот здесь возникает такая же ошибка как и в yolo_tracking репозитории.(((

Получилось!!!
Сначала конвертируем нашу модель в onnx формат запустив скрипт convert.py

 Модель onnx нужно сконвертить с помощью утилиты командой строки:
***trtexec --onnx=osnet_ain_x1_0_triplet_custom.onnx --saveEngine=osnet_ain_x1_0_triplet_custom.engine --fp16 --verbose True***

Затем добавить в файл track.py репы yolo_tracking строчки:
    import tensorrt as trt
    # load all custom plugins
    trt.init_libnvinfer_plugins(None,'')