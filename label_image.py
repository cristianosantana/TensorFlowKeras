import tensorflow as tf, sys
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#import urllib.request

#image_path = sys.argv[1]

__dir__ = "/Workspace/iris-scraping/TensorFlowKeras/"

def start():
    return "starting"

def classifier(image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile("/Workspace/iris-scraping/" + image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                    in tf.gfile.GFile(__dir__ + "retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile(__dir__ + "retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            
            if score >= 0.5:
                data = {}
                data['class'] = human_string
                data['score'] = score
                # Abra o arquivo (leitura)
                arquivo = open(__dir__ + 'resultado.txt', 'r')
                conteudo = arquivo.readlines()

                # insira seu conteúdo
                # obs: o método append() é proveniente de uma lista
                conteudo.append('data: %s;\n' % (data))
                # Abre novamente o arquivo (escrita)
                # e escreva o conteúdo criado anteriormente nele.
                arquivo = open(__dir__ + 'resultado.txt', 'w')
                arquivo.writelines(conteudo)
                arquivo.close()
                return data

if __name__ == "__main__":
    app.start()