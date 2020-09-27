# encoding: utf-8
import sys
sys.path.append("./torchmodels")

import numpy as np

import torch
from torch.backends import cudnn
cudnn.benchmark = True

import tensorflow as tf
from resnet_frompaddle import ResNet152_vd_pytorch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _check_pytorch_tf_model(torch_model_path, tf_model_path, inputshape):
    inputs = torch.randn(1, 3, int(inputshape), int(inputshape))
    model = ResNet152_vd_pytorch(embedding_size=512)
    print(model)
    model.load_state_dict(torch.load(torch_model_path), strict=True)

    model.eval()
    torch_outs = model(inputs)
    #print (torch_outs.shape)

    with tf.Graph().as_default():
        graph_def = tf.compat.v1.GraphDef()
        with open(tf_model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.compat.v1.Session() as sess:
            # init = tf.initialize_all_variables()
            # init = tf.global_variables_initializer()
            # sess.run(init)

            # print all ops, check input/output tensor name.
            # uncomment it if you donnot know io tensor names.

            print('-------------ops---------------------')
            op = sess.graph.get_operations()
            for m in op:
                try:
                    # if 'input' in m.values()[0].name:
                    #     print(m.values())
                    if m.values()[0].shape.as_list()[1] == 2048: #and (len(m.values()[0].shape.as_list()) == 4):
                        print(m.values())
                except:
                    pass
            print('-------------ops done.---------------------')

            input_x = sess.graph.get_tensor_by_name('input:0')  
            outputs = sess.graph.get_tensor_by_name('output:0') 
            output_tf_pb = sess.run(outputs, feed_dict={input_x: to_numpy(inputs)})

    print('output_pytorch = {}'.format(to_numpy(torch_outs)))
    print('output_tf_pb = {}'.format(output_tf_pb))

    np.testing.assert_allclose(to_numpy(torch_outs), output_tf_pb, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with tensorflow runtime, and the result looks good!")


   
if __name__ == '__main__':
    _check_pytorch_tf_model(sys.argv[1], sys.argv[2], sys.argv[3])
