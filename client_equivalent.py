from tritonclient.utils import *
import tritonclient.http as httpclient
import numpy as np
import sys

model_name = "bls_sync"
# shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.random.rand(8, 1, 64, 64).astype(np.float32)
    
    inputs = [
        httpclient.InferInput("INPUT__0", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),

        httpclient.InferInput("MODEL_NAME", [1],
                              np_to_triton_dtype(np.object_)),
    ]
    
    inputs[0].set_data_from_numpy(input0_data)

    # Will perform the inference request on the 'add_sub' model.
    inputs[1].set_data_from_numpy(np.array(['equivalentnet'], dtype=np.object_))

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT__0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT__0")

    
    print("=========='Sazzads Equivalentnet' model result==========")
    print("INPUT__0 ({}) => OUTPUT__0 ({})".format(
        input0_data.shape, output0_data.shape))
    
    
    # if not np.allclose(input0_data, output0_data):
    #     print("BLS sync example error: incorrect sum")
    #     sys.exit(1)

    # if not np.allclose(input0_data - input1_data, output1_data):
    #     print("BLS sync example error: incorrect difference")
    #     sys.exit(1)

        
    print('PASS: BLS Sync')
    sys.exit(0)
