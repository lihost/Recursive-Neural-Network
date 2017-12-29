# Recursive-Neural-Network
Recursive Neural Network for semantic understanding of natural language and associated images. "Grounded Compositional Semantics for Finding and Describing Images with Sentences  ~Socher et al."


# Issues and Resolutions

<b>Issue 1</b> - ``` ImportError: ('The following error happened while compiling the node', InplaceDimShuffle{x}(TensorConstant{0.0}), '\n', '/home/sam/.theano/compiledir_Linux-4.8--generic-i686-with-debian-stretch-sid-i686-2.7.13-32/tmpVPNwFe/0abe19206004f55475830d020f9a6684.so: undefined symbol: _ZdlPvj') ```
<br><b>Resolution</b> - ``` export LD_PRELOAD=/usr/lib/i386-linux-gnu/libstdc++.so.6 ```


<b>Issue 2</b> - Related  to float64 bit
<br><b>Resolution</b> - ``` echo -e "\n[global]\nfloatX=float32\n" >> ~/.theanorc ```



# Preparing and running code
## DT-RNN
1. Run ```dparse_to_dtree_dtr.py``` (Provide required configs)
	* ```SENT_FILE``` - Sentence text file
	* ```RAW_PARSES_FILE``` - Filename for raw parses of dependency tree
	* ```STANFORD_LEXPARSER``` - Location of stanford lexical parser
	* ```FINAL_SPLIT_FILE``` - Final split filename
2. Run ```word2vec_gen.py``` (Provide required text file in directory location)
	* ```SENT_DIR``` - Keep sentence text file at this location
	* ```SAVE_MODEL``` - Word2Vec generated embeddings are saved as this file
3. Run ```single_process_dtr.py``` (Provide required configs)
	* ```FINAL_SPLIT_FILE``` - file location generated in step 1
	* ```W2V_FILE``` - Word2Vec embedding file generated in step 2
	* ```SAVE_NPY``` - Filename for saving complete dependency tree for sentences
	


## Image-Processing
1. Create 3 separate image datasets for train, test and dev by using following command.
	```sh 
	find <Flicker8k_Dataset/> | grep -F -f <image_list_train> | xargs cp -t <imageset_train/>
	```
2. Run ```image_process.py``` (Provide required configs)
	* ```IMAGE_SET_DIR``` - folder location where images(for train, test, dev) are kept.
	* ```IMAGE_FILENAMES``` - Text file mentioning image names(for train, test, dev) as provided by flickr.
	* ```SAVED_IMAGE_VEC``` - Filename for saving individual image vectors(train, test, dev).


## Max-Margin objective - Main training and evaluation
1. Run ```data_preprocess.py``` (Provide required configs)
	* Provide sentence vectors(train, test, dev) and image vectors(train, test, dev) files which we generated from DT-RNN and Image-Processing steps respectively.
	
	* Just for the sake of running training, I have already provided required vector files.
  
  
  
  
