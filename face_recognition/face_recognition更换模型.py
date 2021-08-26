# %%
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
from config import configs
import tensorflow as tf
import os


DIRECTION = []
NAME = []
SMILIES = []
SUNGLASSES = []



def read_img(img_path):
    im = Image.open(img_path).convert("L") # 30*32
    arr = np.array(list(im.getdata())) # getdata 返回展平的像素值序列
    return arr


pass

# %%
def get_imgs_path(dir_path):
    if not os.path.exists(dir_path):
        print("Error:IMGPATH is path not exists")
        exit()
    img_list = []
    list_dir = os.listdir(dir_path)
    for dir in list_dir:
        current_dir_path = dir_path + dir + '/'
        # 获取当前目录下的图片名
        dir_img_name_list = os.listdir(current_dir_path)
        for img_name in dir_img_name_list:
            strs = img_name.split('_')
            if not strs[0] in NAME:
                NAME.append(strs[0])
            if not strs[1] in DIRECTION:
                DIRECTION.append(strs[1])
                pass
            if not strs[2] in SMILIES:
                SMILIES.append(strs[2])
            if not strs[3] in SUNGLASSES:
                SUNGLASSES.append(strs[3])
            img_list.append(current_dir_path + img_name)

    print(DIRECTION)
    print(NAME)
    print(SMILIES)
    print(SUNGLASSES)
    return img_list

# %%
pass

# %%
def get_imgs(img_paths):
    imgs = []
    names = []
    directions = []
    smilies = []
    sunglasses = []
    # 取出要处理的img
    # print(img_paths)
    for img_path in img_paths:

        img_val = read_img(img_path)
        imgs.append(img_val)
        face_name = np.ones(len(NAME)) / 10
        face_direction = np.ones(len(DIRECTION)) / 10
        face_smilies = np.ones(len(SMILIES)) / 10
        face_sunglasses = np.ones(len(SUNGLASSES)) / 10

        strs = img_path.split('/')[-1].split('_')
        face_name[NAME.index(strs[0])] = 0.9
        face_direction[DIRECTION.index(strs[1])] = 0.9
        face_smilies[SMILIES.index(strs[2])] = 0.9
        face_sunglasses[SUNGLASSES.index(strs[3])] = 0.9
        
        names.append(face_name)
        directions.append(face_direction)
        smilies.append(face_smilies)
        sunglasses.append(face_sunglasses)
    pass

    return_val = {
        'imgs': np.array(imgs),
        'face_name': np.array(names),
        'face_direction': np.array(directions),
        'face_smilies': np.array(smilies),
        'face_sunglasses': np.array(sunglasses)
    }
    return return_val
# 可视化
def plot_image(i, predictions_array, true_label, img, LABEL):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(LABEL[predicted_label],
                                100*np.max(predictions_array),
                                LABEL[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label, LABEL):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(len(LABEL)))
  plt.yticks([])
  thisplot = plt.bar(range(len(LABEL)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# %%

def main():
    # 更换预测特征，两个一起改
    train_type = 'face_direction'
    LABEL = DIRECTION
    img_name_list = get_imgs_path(configs['IMGPATH'])
    random.shuffle(img_name_list)
    train_data = get_imgs(img_name_list[:400])
    test_data = get_imgs(img_name_list[400:])

    train_data_size = len(train_data['imgs'])
    test_data_size = len(test_data['imgs'])
    print('训练集大小: ' + str(train_data_size))
    print('测试集大小: ' + str(test_data_size))

    train_imgs = np.reshape(train_data['imgs'],(400,30,32))
    test_imgs = np.reshape(test_data['imgs'],(224,30,32))
    train_label_index = np.argmax(train_data[train_type] ,axis = 1)
    test_label_index = np.argmax(test_data[train_type] ,axis = 1)
    # 像素归一化
    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0
    # 搭建模型
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 32)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(LABEL))])
    # 编译
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    # 训练
    model.fit(train_imgs, train_label_index, epochs=100)
    plt.show()
    # 测试
    test_loss, test_acc = model.evaluate(test_imgs,  test_label_index, verbose=2)

    print('\nTest accuracy:', test_acc)
    # 把输出转化成概率
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_imgs)
    for i in range(10):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plot_image(i, predictions[i], test_label_index, test_imgs, LABEL)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions[i], test_label_index,LABEL)
    plt.show()      
                               
    
# %%
if __name__ == '__main__':
    main()

# %%
