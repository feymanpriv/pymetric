# encoding: utf-8
"""
    convert torch 2 tf via onnx
"""

import warnings
warnings.filterwarnings('ignore')  # Ignore all the warning messages in this tutorial

from onnx_tf.backend import prepare
import tensorflow as tf
from PIL import Image
import torchvision.transforms as transforms

import onnx
import numpy as np
import torch
from torch.backends import cudnn
import io

cudnn.benchmark = True


def export_tf_model(model: torch.nn.Module, tensor_inputs: torch.Tensor):
    """
    Export a model via ONNX.
    Arg:
        model: a tf_1.x-compatible version of detectron2 model, defined in caffe2_modeling.py
        tensor_inputs: a list of tensors that caffe2 model takes as input.
    """
    assert isinstance(model, torch.nn.Module)

    # Export via ONNX
    print("Exporting a {} model via ONNX ...".format(type(model).__name__))
    predict_net = _export_via_onnx(model, tensor_inputs)
    print("ONNX export Done.")


def _export_via_onnx(model, inputs):
    def _check_val(module):
        assert not module.training

    model.apply(_check_val)

    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                # verbose=True, 
                export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    torch.onnx.export(model,
                      inputs,  
                      "convertmodel.onnx",  
                      export_params=True,  
                      opset_version=10,  
                      do_constant_folding=True,  
                      input_names=['input'],  
                      output_names=['output'],  
                      dynamic_axes={'input': {0: 'batch_size'},  
                                    'output': {0: 'batch_size'}}
                      )


def export_tf():
    onnx_model = onnx.load("middlemodel.onnx")
    # Convert ONNX Model to Tensorflow Model
    tf_rep = prepare(onnx_model, strict=False)  
    print(tf_rep.inputs)  # Input nodes to the model
    print('-----')
    print(tf_rep.outputs)  # Output nodes from the model
    print('-----')
    print(tf_rep.tensor_dict)  # All nodes in the model
    # """

    # install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
    # Reference https://github.com/onnx/onnx-tensorflow/issues/167
    # tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'

    # debug, here using the same input to check onnx and tf.
    # output_onnx_tf = tf_rep.run(to_numpy(img))
    # print('output_onnx_tf = {}'.format(output_onnx_tf))
    # onnx --> tf.graph.pb
    # tf_pb_path = 'tf_graph.pb'
    # tf_rep.export_graph(tf_pb_path)

    return tf_rep


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _check_pytorch_tf_model(model: torch.nn.Module, tf_graph_path: str):
    img = Image.open("demo_imgs/dog.jpg")

    resize = transforms.Resize([384, 128])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img.unsqueeze_(0)
    torch_outs = model(img)

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(tf_graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            # init = tf.initialize_all_variables()
            # init = tf.global_variables_initializer()
            # sess.run(init)

            # print all ops, check input/output tensor name.
            # uncomment it if you donnot know io tensor names.
            '''
            print('-------------ops---------------------')
            op = sess.graph.get_operations()
            for m in op:
                try:
                    # if 'input' in m.values()[0].name:
                    #     print(m.values())
                    if m.values()[0].shape.as_list()[1] == 2048: 
                        print(m.values())
                except:
                    pass
            print('-------------ops done.---------------------')
            '''
            input_x = sess.graph.get_tensor_by_name('input.1:0')  # input
            outputs = sess.graph.get_tensor_by_name('502:0')  # 5
            output_tf_pb = sess.run(outputs, feed_dict={input_x: to_numpy(img)})

    print('output_pytorch = {}'.format(to_numpy(torch_outs)))
    print('output_tf_pb = {}'.format(output_tf_pb))

    np.testing.assert_allclose(to_numpy(torch_outs), output_tf_pb, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with tensorflow runtime, and the result looks good!")


