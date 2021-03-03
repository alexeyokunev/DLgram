# DLgram
this is a telegram interface to training and inference on Cascade-Mask RCNN network

👍TRAIN
1. To train net on the crop annotate part of your image in labelme using polygons. Only two label classes must be used
- "crop" specify   the region(s) of image used for training. crops are to be nearly rectangular. Probably crops are to be of the same size for better traning.
- "label" specify the objects to recognize. All objects inside each of the crops are to be annotated, including those cross-sected by the crops margins. 
2. download annotated labelme file into the group without caption
3. wait
4. write down or remember the name of the trained net to use it further for inference. the net name coincides with the name of the labelme file without extention.
5. download labelme file sent by bot to see pretrained net prediction on the whole image 

👌INFERENCE
1. download the image file into the group
2. make sure "compressed" is not checked
3. type the name of a pretrained net in the caption
4.  download labelme file sent by bot to see pretrained net prediction

❤️STATISTICS
1. open Labelme program, okunev version
2. open predicted labelme json file
3. press Calculate stat button
4. search for the *.csv file with statistics nearby labelme json file w. it has the same name as the source labelme json

🤞LABELME FOR WINDOWS
https://drive.google.com/file/d/1uxgeTh5sjLA6-mbVC1hq7v7UWmz7as_S/view?usp=sharing
