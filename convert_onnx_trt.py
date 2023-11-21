import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [22, 3, 256, 128]):

    """
    This is the function to create the TensorRT engine
    Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_serialized_network(network, config)
        
        #context = engine.create_execution_context()
        #inspector = engine.create_engine_inspector()
        #inspector.execution_context = context # OPTIONAL
        #print(inspector.get_layer_information(0, trt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.
        #print(inspector.get_engine_information(trt.LayerInformationFormat.JSON)) # Print the information of the entire engine.
       
        return engine

def save_engine(engine, file_name):
   #buf = engine.serialize()
   with open(file_name, 'wb') as f:
    f.write(engine)
       
def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

if __name__ == '__main__':
    onnx_path = 'osnet_ain_x1_0_triplet_custom.onnx'
    engine_path = 'osnet_ain_x1_0_triplet_custom1.engine'
    engine = build_engine(onnx_path)
    save_engine(engine, engine_path)
    
    model = load_engine(engine, engine_path)