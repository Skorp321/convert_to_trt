import torchreid
import torch
import torch_tensorrt
from torch2trt import torch2trt

model_name = 'osnet_ain_x1_0'
loss = 'triplet'
use_gpu = torch.cuda.is_available()

def main():
    
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=946,
        loss=loss,
        pretrained=False,       
    )

    torchreid.utils.load_pretrained_weights(model, 'models/osnet_ain_x1_0_triplet_custom.pt')
    
    model = model.eval().cuda()  # torch module needs to be in eval (not training) mode

    inputs = torch.randn((32, 3, 256, 128)).cuda()
    input_names = [ "input" ]
    output_names = [ "output" ]

    #export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    #export_output = torch.onnx.dynamo_export(
    export_output = torch.onnx.export(    
        model,
        inputs,
        "my_dynamic_model.onnx",
        #verbose=True, 
        input_names=input_names, 
        output_names=output_names)
        #export_options=export_options)
    #export_output.save("my_dynamic_model.onnx")
    
if __name__ == "__main__":
    main()