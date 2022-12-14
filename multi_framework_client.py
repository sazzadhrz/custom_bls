from tritonclient.utils import *
import tritonclient.http as httpclient
import numpy as np
import sys
import cv2

model_name = "tf_torch_bls"
# shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    # input0_data = np.random.rand(4, 256, 512, 1).astype(np.float32)
    # input0_data = cv2.imread('./buet.png').astype(np.float32)
    input0_data = cv2.imread('./typewriter.jpg').astype(np.float32)
    input0_data = np.expand_dims(input0_data, axis=0)
    
    inputs = [
        httpclient.InferInput("image", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),

        httpclient.InferInput("MODEL_NAME", [1],
                              np_to_triton_dtype(np.object_)),
    ]
    
    inputs[0].set_data_from_numpy(input0_data)

    # Will perform the inference request on the 'add_sub' model.
    inputs[1].set_data_from_numpy(np.array(['autonoise'], dtype=np.object_))

    outputs = [
        httpclient.InferRequestedOutput("word_segments"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("word_segments")
    # cv2.imwrite('cleaned_from_triton.png', output0_data*255.0)
    # print('type(result)', type(result))
    # print('type(output0_data)', type(output0_data))

    print(output0_data)

    # print(response.get_response()['outputs'][0])

    
    print("=========='AutoNoise CycleGan' model result==========")
    print("Input (input_3) ({}) => OUTPUT (conv2d_17) ({})".format(
        input0_data.shape, output0_data.shape))
    
    # ######################### second call #########################
    # inputs[0].set_data_from_numpy(output0_data)

    # response = client.infer(model_name,
    #                         inputs,
    #                         request_id=str(1),
    #                         outputs=outputs)

    # result = response.get_response()
    # output0_data = response.as_numpy("conv2d_17")

    # # print(response.get_response()['outputs'][0])

    
    # print("=========='AutoNoise CycleGan' model result==========")
    # print("Input (input_3) ({}) => OUTPUT (conv2d_17) ({})".format(
    #     input0_data.shape, output0_data.shape))

        
    print('PASS: AutoNoise BLS Sync')
    sys.exit(0)
