# encoding: utf-8
import sys

import warnings
warnings.filterwarnings('ignore')  # Ignore all the warning messages in this tutorial

from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import onnx


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export(onnx_model_path, tf_pb_path, inputshape):
    onnx_model = onnx.load(onnx_model_path)
    print("load success")
    onnx.checker.check_model(onnx_model)
    print("check success")
    tf_rep = prepare(onnx_model, strict=True) # Import the ONNX model to TF
    print(tf_rep.inputs)  # Input nodes to the model
    print('-----')
    print(tf_rep.outputs)  # Output nodes from the model
    print('-----')
    print(tf_rep.tensor_dict)  # All nodes in the model

    dummy_inputs = torch.randn(1, 3, int(inputshape), int(inputshape))
    output_onnx_tf = tf_rep.run(to_numpy(dummy_inputs))
    print('output_onnx_tf = {}'.format(output_onnx_tf))
    # onnx --> tf.graph.pb
    tf_rep.export_graph(tf_pb_path)


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], sys.argv[3])
