############
#find Flicker8k_Dataset/ | grep -F -f image_list | xargs cp -t imageset/
#find Flicker8k_Dataset/ | grep -F -f image_list | xargs cp -t imageset/
#find Flicker8k_Dataset/ | grep -F -f image_list | xargs cp -t imageset/
############



import tensorflow as tf
import numpy as np
import os
import glob
#Input : set of training images  (imgName)
#Output:  2048 Dimensional vector (Bottleneck_tensor)

IMAGE_SET_DIR = '/imageset_dev'
IMAGE_FILENAMES = 'Flickr_8k.devImages.txt'
SAVED_IMAGE_VEC = 'img_vec_dev5.npy'

def create_graph():
    """
    Creates a graph from saved GraphDef file and returns a saver.
    """
# Creates graph from saved graph_def.pb.
# modelFullPath : Inception model path
    modelFullPath = 'classify_image_graph_def.pb'

    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:

        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        _ = tf.import_graph_def(graph_def, name='')



def gen_image_vector(imgName):
    
    
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

    #image_data = tf.gfile.FastGFile(imgName, 'rb').read()
    with tf.Session() as sess:
        bottleneck_tensor = sess.graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
        image_data = tf.gfile.FastGFile(imgName, 'rb').read()
        BottleneckTensor = sess.run(bottleneck_tensor, {'DecodeJpeg/contents:0': image_data})

    return BottleneckTensor[0]

            

if __name__ == '__main__':
    create_graph()
    print('Graph created...\n')
    
    img_dir = os.getcwd() + IMAGE_SET_DIR
    
    print('creating image vectors...\n')

    vectors = []
    count=1
    with open(IMAGE_FILENAMES, 'r') as f:
	for name in f:    
            f = glob.glob(os.path.join(img_dir, name.strip()))
		
            for i in range(5):
                vectors.append(gen_image_vector(f[0]))
                print(count," : ",i)
		#print(gen_image_vector(f[0]))
            count += 1    
    
    #print(vectors)    
    print('saving image vectors...\n')
    np.save(SAVED_IMAGE_VEC, vectors)

    vec = np.load(SAVED_IMAGE_VEC)
    print(vec.shape)
