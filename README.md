# INSTRUCTION

👍TRAIN
1. To train the deep neural network annotate several objects on your image in labelme using "polygon" shape. 
2. download annotated labelme file into the group (if you are dare enough use caption as a commands to the network
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

TIPS: USE OF COMMANDS
on training:

load the file annotated in labelme. the crop class is NOT currently in use!
we use commands during training

-a 8 - (augmentation) turns each marked up object into 8 (or how many you specify) several modified objects. It is necessary for high-quality training of the network. empirical parameter. rule of thumb - the number of particles of each class is not less than 10, taking into account augmentation. For example, 4 class particles are marked, -a 4, total 4 * 4 = 16 class particles in training

- e 100 - number of epochs. during each epoch, the network sees all particles of all classes. If the network has a lot of particles, it needs fewer epochs. If the number of particles in a class is about 10-20, then good results are obtained at 100 epochs. you can bet more, but it loads the server a lot. while one trains - no one else can do anything.
-r 1.5 - now the program itself chooses the crop around the object. 1.5 means that the crop size will be 1.5 times the size of the object. empirical parameter

The rest of the parameters are NOT needed during training!

on inference

the parameters that were used during the training are NOT needed!

-n @ #% $ # - the name of the EXISTING trained network with which you want to carry out the recognition. The bot will write this name after the workout is over. It must be memorized (written down), otherwise it will have to be trained again.

-t 0.3 (thresh) trigger threshold. Numbers from 0 to 1. If close to zero, all network guesses will be turned on, most of which are garbage. If it is close to 1, the network produces only verified guesses, for example, only the particle that coincides with the original, or nothing

-m 0 (merge) if there were several classes, then if the value is 0, the classes will be saved. If the value is 1, all network guesses will be collected into one class and given to the user. For example, you need to count small and large separately. And then we see that they overlap a lot and the network cannot distinguish between them. Then put -m 1.

ИСПОЛЬЗОВАНИЕ КОМАНД

по тренировке:

загружаем размеченный в labelme файл. класс crop сейчас НЕ используется!
при тренировке используем команды

-a 8  - (augmentation) превращает каждый размеченный объект в 8 (или сколько укажете) несколько измененных объектов. Нужно для качественной тренировки сети. эмпирический параметр. rule of thumb - число частиц каждого класса не меньше 10 с учетом augmentation. Например, размечено 4 частицы класса, -a 4, итого 4*4 = 16 частиц класса на тренировке

- e 100 - число эпох. в течении каждой эпозхе сеть видит все частицы их всех классов. Если у сети много частиц - ей нужно меньше эпох. Если число частиц в классе около 10-20, то неплохие результаты получаются при 100 эпохах. можно ставить больше, но это сильно грузит сервер. пока один тренирует - больше никто ничего не может делать.
-r 1.5 - сейчас программа сама выбирает crop вокруг объекта. 1.5 означает, что размер кропа будет в 1.5 раз больше размера объекта. параметр эмпирический

Остальные параметры при тренировке НЕ нужны!

по инференсу

праметры, которые использованы при тренировке НЕ нужны!

-n  @#%$# - название СУЩЕСТВУЮЩЕЙ натренированной сети, с помощью которой хотите провести распознавание. Это название бот напишет после того, как тренировка окончениа. Его надо запомнить (записать), иначе снова придется тренировать.

-t 0.3 (thresh) порог срабатывания. Числа от 0 до 1. Если близко к нулю - включабтся все догадки сети, большинство из которых - мусор. Если близко к 1 - сеть выдает только проверенные предположения, например, только ту частицу, которая совпадает с исходной или ничего

-m 0 (merge) если было несколько классов, то при значении 0 классы будут сохраняться. При значении 1 все предположения сети будут собраны в один класс и выданы пользователю. Например, нужно по отдельности посчитать большие и маленькие. А потом мы видим, что они сильно перекрывыаются и сеть не может их различить. Тогда ставим -m 1. 
