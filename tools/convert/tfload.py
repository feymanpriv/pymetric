# encoding: utf-8
import tensorflow as tf

import numpy as np
import io

from tensorflow import keras

def load_tf2_saved_model(model_save_path):
    loaded = keras.models.load_model(model_save_path)
    print(list(loaded.signatures.keys()))  # ["serving_default"]
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)

#export_dir = "./tf2"
with tf.Graph().as_default():
    graph_def = tf.compat.v1.GraphDef()
    with open("yangmin09_testr50_tf_graph.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        print("import success")
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
                if 'input' in m.values()[0].name:
                    print(m.values())
                if m.values()[0].shape.as_list()[1] == 2048: #and (len(m.values()[0].shape.as_list()) == 4):
                    print(m.values())
            except:
                pass
        print('-------------ops done.---------------------')
        
        input_x = sess.graph.get_tensor_by_name('input:0')  # input
        outputs = sess.graph.get_tensor_by_name('output:0')  # 5
        print (outputs)
        #output_tf_pb = sess.run(outputs, feed_dict={input_x: to_numpy(img)})
        print(output_tf_pb)
        #builder.add_meta_graph_and_variables(sess,
                                   #[tf.compat.v1.saved_model.tag_constants.TRAINING])
        print("save success")


