import argparse
import os
import os.path as osp
import json
import subprocess
import requests

import telebot

import PIL.Image
import io


class Bot:
    DEFAULT_ARGS = dict(n='stock',
                             t=0.01,
                             e=20,
                             a=8,
                             r=1.5,
                             m=0
                             )
    def __init__(self, TOKEN):
        self.TOKEN = TOKEN
        self.bot = telebot.TeleBot(self.TOKEN, parse_mode=None)
        self.chat_id = None
        self.is_busy = False
        self.name = None
        self.train_json = None
        self.test_json = None
        self.img = None
        self.path_to_img = None
        self.text = None
        self.msg_json = None
        self.args = self.DEFAULT_ARGS.copy()
        self.args_str = ''
        
    def _get_file(self, doc):
        file_id = doc['file_id']
        file_info = self.bot.get_file(file_id)
        file = requests.get(f'https://api.telegram.org/file/bot{self.TOKEN}/{file_info.file_path}')
        
        if file.status_code == 200:
            return file
        else:
            self.bot.send_message(self.chat_id, f'error with message {self.msg_json}')
            return None
           
    def _handle_doc(self, message):
        self.msg_json = message.json
        doc = self.msg_json['document']
        self.file_name = doc['file_name']
        self.busy_check(doc)
        
    def _handle_photo(self, message):
        self.msg_json = message.json
        doc = self.msg_json['photo'][-1]
        self.file_name = doc['file_unique_id'] + '.jpg'
        self.busy_check(doc)
        
    def busy_check(self, doc):
        if self.is_busy:
            self.bot.send_message(self.chat_id, 'I\'m busy with previous task. Please repost your \
                                                 data after I will have finished.')
        else:
            self.is_busy = True
            self.chat_id = self.msg_json['chat']['id']
            self._main_handler(doc)
                
    def _main_handler(self, doc):
        file = self._get_file(doc)
        
        file_name_base = ''.join(self.file_name.split('.')[:-1])
        file_name_ext = self.file_name.split('.')[-1]
        
        if file_name_ext == 'json':
            self.args['n'] = ''.join(file_name_base)
            net = self.args['n']
            model_dir = osp.join('../data', net)
            if osp.isdir(model_dir):
                self.bot.send_message(self.chat_id, 
                                      f'you\'ve already trained the dataset. ask admin to help retrain')
            else:
                os.makedirs(model_dir)
                self.bot.send_message(self.chat_id, 
                                      'Wait, please. I\'m training the model ...')
                # save train json with labelme label
                path_to_train_json = osp.join(model_dir, self.file_name)
                path_to_train_json = path_to_train_json.replace('.json', '_labelme.json')
                self.train_json = file.content.decode()
                while type(self.train_json) == str:
                    self.train_json = json.loads(self.train_json)
                with open(path_to_train_json, 'w') as f:
                    json.dump(self.train_json, f)
                
                if 'caption' in self.msg_json.keys():
                    self._set_args(self.msg_json['caption']) 
                    
                self.args_str = f'python3 train.py {self._get_args_str()}' 
                self.p = subprocess.run(self.args_str.split(), capture_output=True)
                if self.p.returncode == 0:
                    self.bot.send_message(self.chat_id, 'Training completed')
                    test_dir = osp.join('../data', net, 'test')
                    path_to_res = osp.join(test_dir, net + '.' + 'json')
                    doc = open(path_to_res, 'rb')
                    self.bot.send_document(self.chat_id, doc, 
                                           caption=f'Type \"{net}\" in the image caption to use \
                                                     the net for inference')

                    dst_stat_name = osp.join(test_dir, net + '.csv')
                    with open(dst_stat_name, 'r') as f:
                        stat = json.load(f)

                    res_img_name = [name for name in os.listdir(test_dir) if '_inf.' in name][0]
                    path_to_res_img = osp.join(test_dir, res_img_name)
                    photo = open(path_to_res_img, 'rb')
                    self.bot.send_photo(self.chat_id, photo, caption=f'found {stat}')

                else:
                    print(self.p)
                    self.bot.send_message(self.chat_id, f'error with processing {self.train_json}')

        if file_name_ext in ('bmp', 'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'png'):
            if 'caption' in self.msg_json.keys():
                self._set_args(self.msg_json['caption'])
            
            net = self.args['n']
            test_dir = osp.join('../data', net, 'test')
            if not osp.isdir(test_dir):
                self.bot.send_message(self.chat_id, f'there is no trained net {net}')
            else:
                self.bot.send_message(self.chat_id,
                                      f'Wait, please. I\'m testing the model {net}...')
                self.img = PIL.Image.open(io.BytesIO(file.content))
                path_to_img = osp.join(test_dir, self.file_name)
                self.img.save(path_to_img)
             
                self.args_str = f'python3 inference.py {path_to_img} {self._get_args_str()}'
                self.p = subprocess.run(self.args_str.split(), capture_output=True)
                if self.p.returncode == 0:
                    dst_stat_name = osp.join(test_dir, file_name_base + '.csv')
                    with open(dst_stat_name, 'r') as f:
                        stat = json.load(f)

                    path_to_res_img = osp.join(test_dir, file_name_base + '_inf.' + file_name_ext)
                    photo = open(path_to_res_img, 'rb')
                    self.bot.send_photo(self.chat_id, photo, caption=f'found {stat}')
    
                    path_to_res_json = osp.join(test_dir, file_name_base + '.' + 'json')
                    doc = open(path_to_res_json, 'rb')
                    self.bot.send_document(self.chat_id, doc, caption=f'predicted using {self.args}')

                else:
                    print(self.p)
                    self.bot.send_message(self.chat_id, f'error with processing {self.file_name}')
        # set args to default values
        self._set_args('')
        # ready to accept new requests
        self.is_busy = False

    def _handle_cmd(self, message):
        self.msg_json = message.json
        self.chat_id = self.msg_json['chat']['id']
        if self.msg_json['text'] == '/stop':
            self.bot.send_message(self.chat_id, "My activity stopped. Ask admin to restart, if you need me")
            self.stop()
            print('bot stopped')
        elif self.msg_json['text'] == '/help':
            with open('instruction.txt', 'r') as f:
                help_msg = f.read()
            self.bot.send_message(self.chat_id, help_msg[:2048])
            print('help sent')
        elif self.msg_json['text'] == '/args':
            self.bot.send_message(self.chat_id, f'set args {self.args}')
            
    def _handle_text(self, message):
        self.msg_json = message.json
        self.chat_id = self.msg_json['chat']['id']
        
    def _set_args(self, text):
        if text.startswith('-'):
            args = text.strip().split()
            for i in range(len(args)//2):
                key = args[i*2]
                key = key[1:]
                value = args[i*2+1]
                if key in self.args.keys():
                    self.args[key] = value
        else:
            self.args = self.DEFAULT_ARGS.copy()
        self.bot.send_message(self.chat_id, f'set args {self.args}')
    
    def _get_args_str(self):
        self.args_str = ''
        for key in self.args.keys():
            self.args_str += ' -' + key + ' ' + str(self.args[key])  
        return self.args_str
        
    def run(self):
        self.bot.polling()
        
    def stop(self):
        self.bot.stop_bot()   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("TOKEN", help="telegram bot token")

    args = parser.parse_args()
    bot = Bot(args.TOKEN)
    
    @bot.bot.message_handler(commands=['help', 'args'])  # 'stop',
    def handle_cmd(message):
        bot._handle_cmd(message)

    @bot.bot.message_handler(content_types=['document'])
    def handle_doc(message):
        bot._handle_doc(message)
    
    @bot.bot.message_handler(func=lambda message: True)
    def handle_text(message):
        bot._handle_text(message) 

    @bot.bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        bot._handle_photo(message) 
    
    bot.run()


if __name__ == '__main__':
    main()
