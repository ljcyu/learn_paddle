# 导入需要的包
import zipfile
import random
import numpy as np
import json
from PIL import Image
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import os, Augmentor
import shutil, glob
SEED = 1000
def unzip_data(src_path, target_path):
    '''
    解压原始数据集
    '''

    if (not os.path.isdir(target_path)):
        print(f"源文件地址：{src_path}", f"解压目标目录：{target_path}")
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")
    __MACOSX = Path(target_path) / '__MACOSX'
    if __MACOSX.is_dir():
        shutil.rmtree(__MACOSX)


def data_reader(df):
    '''
    自定义data_reader
    '''

    def reader():
        for img_path, _, lbl in df.itertuples(index=False):
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((64, 64), Image.BILINEAR)
            img = np.array(img).astype('float32')
            img = img.transpose((2, 0, 1))  # HWC to CHW 
            img = img / 255  # 像素值归一化
            yield img, int(lbl)

    return reader


# 此处代码主要是消除原始 4 通道图片的影响
def proc_img(src):
    for root, dirs, files in os.walk(src):
        for file in files:
            src = os.path.join(root, file)
            # print(src)
            img = Image.open(src)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(src)

# ### 划分训练集测试集

# 图片增强前原始的train和eval
# train_params['target_path']
def gen_train_eval_before(target_path):
    # 训练数据文件夹
    targetPath = Path(target_path)
    class_dirs = sorted(targetPath.glob("*"))
    print(class_dirs)
    # 获取数据 metadata
    lst_data = []
    for i, class_dir in enumerate(class_dirs):
        lst_path = list(class_dir.glob("*.jpg"))
        lst_gemName = [p.parent.name for p in lst_path]
        # zip当前路径、名称以及该路径下的宝石类别,
        # lst_path、lst_genName都是一个list，所以类别也需要是一个list
        lst_data.extend(zip(map(str, lst_path), lst_gemName, [i] * len(lst_path)))

    # 构建数据 dataframe 并且打乱
    df_data = pd.DataFrame(lst_data, columns=['gem_path', 'gem_name', 'lbl']).sample(frac=1, replace=False,
                                                                                     random_state=SEED)
    # 分割 traing,validation 数据集
    train_data, eval_data = train_test_split(df_data, test_size=0.1, random_state=42)
    return train_data, eval_data


# eval_data中的图片移动到另一个目录，然后可以对剩下图片进行增强
# 就不用担心eval中数据被增强的问题
# eval.txt文件移动到另一个目录，然后对train.txt中文件进行数据增强
def move_eval_imgs(eval_data):
    for row in eval_data[['gem_path', 'dest_path']].itertuples():
        src_path = row[1]
        dest_path = row[2]
        tmp_ = Path(dest_path)
        if not tmp_.exists(): tmp_.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dest_path)


def gen_eval_txt(eval_data, eval_txt_path):
    with open(eval_txt_path, 'w') as f:
        for row in eval_data[['dest_path', 'lbl']].itertuples():
            dest_path = str(row[1])
            label = str(row[2])
            f.write('{}\t{}\n'.format(dest_path, label))


# 拷贝评估集文件到另一个目录
def proc_eval(eval_data, eval_txt):
    # apply
    eval_data.loc[:, 'dest_path'] = eval_data.loc[:, 'gem_path'].apply(
        lambda x: str(x).replace('dataset', 'dataset_eval'))
    eval_data.head(5)
    move_eval_imgs(eval_data)
    gen_eval_txt(eval_data, eval_txt)


def aug(augment_path, gen_img_count, img_root):
    if not os.path.exists(augment_path):  # 控制不重复增强数据
        for root, dirs, files in os.walk(img_root, topdown=False):
            for name in dirs:
                path_ = os.path.join(root, name)
                if '__MACOSX' in path_: continue
                print('数据增强：', os.path.join(root, name))
                print('image：', os.path.join(root, name))

                p = Augmentor.Pipeline(os.path.join(root, name), output_directory='output')
                p.rotate(probability=0.6, max_left_rotation=2, max_right_rotation=2)
                p.zoom(probability=0.6, min_factor=0.9, max_factor=1.1)
                p.random_distortion(probability=0.4, grid_height=2, grid_width=2, magnitude=1)
                p.flip_left_right(probability=0.3)
                p.flip_top_bottom(probability=0.3)
                p.crop_random(probability=0.3, percentage_area=0.8)
                p.greyscale(probability=0.2)
                p.random_brightness(probability=0.2, min_factor=0.8, max_factor=1.2)

                count = gen_img_count - len(glob.glob(pathname=path_ + '/*.jpg'))
                p.sample(count, multi_threaded=False)
                p.process()

        print('将生成的图片拷贝到正确的目录')
        tmp_dirs = Path(img_root).iterdir()
        # print(tmp_dirs)
        for dir_ in tmp_dirs:
            src_path = dir_ / 'output'
            dest_path = augment_path + "/" + dir_.name
            # print(src_path,dest_path)
            shutil.move(str(src_path), dest_path)
        print('完成数据增强')


def gen_train_after_aug(augment_path):
    targetPath_aug = Path(augment_path)
    class_dirs_aug = sorted(targetPath_aug.glob("*"))
    lst_data = []
    for i, class_dir in enumerate(class_dirs_aug):  # 遍历增强后的数据
        lst_path = list(class_dir.glob("*.jpg"))
        img = [p.name for p in lst_path]
        lst_gemName = [p.parent.name for p in lst_path]
        lst_data.extend(zip(map(str, lst_path), img, lst_gemName, [i] * len(lst_path)))

    # 构建数据 dataframe 并且打乱
    train_data = pd.DataFrame(lst_data, columns=['gem_path', 'img', 'gem_name', 'lbl']).sample(frac=1, replace=False,
                                                                                               random_state=SEED)
    train_data.head(10)
    return train_data


# 重新产生train.txt
# 生成标签
def gen_train_txt(train_data, train_txt_path):
    with open(train_txt_path, 'w') as f:
        for row in train_data[['gem_path', 'lbl']].itertuples():
            dest_path = str(row[1])
            label = str(row[2])
            f.write('{}\t{}\n'.format(dest_path, label))


def info(train_data, eval_data):
    print("== 数据集总体情况:总类别数", train_data.lbl.max() + 1)
    print("== 训练集不同类别的样本数：")
    print(train_data.gem_name.value_counts())
    print(f"== 训练集样本数：{len(train_data)}", f"验证集样本数：{len(eval_data)}")


"""
augment_path: 增强文件存放路径
src_path:数据集压缩文件名
target_path
eval_txt
train_txt
gen_img_count
"""


def gen_train_eval(src_path, target_path, augment_path, train_txt, eval_txt, img_enhance,gen_img_count):
    unzip_data(src_path, target_path)
    proc_img(target_path)
    train_data, eval_data = gen_train_eval_before(target_path)
    proc_eval(eval_data, eval_txt)
    if img_enhance:
        aug(augment_path, gen_img_count, target_path)
        train_data = gen_train_after_aug(augment_path)
    else:
        train_data = gen_train_after_aug(target_path)
    gen_train_txt(train_data, train_txt)
    train_data.reset_index(drop=True, inplace=True)
    eval_data.reset_index(drop=True, inplace=True)
    info(train_data, eval_data)
    return train_data, eval_data


def demo():

    random.seed(SEED)
    np.random.seed(SEED)
    dataset_prefix = 'test'  # 数据集名
    work_dir = '../../first_paddle/work/'
    train_params_path = work_dir  + dataset_prefix + '_params.json'  # 保存train_params
    train_df_json = work_dir + dataset_prefix + '_train_df.json'
    eval_df_json = work_dir + dataset_prefix + '_eval_df.json'

    train_params = {
        'input_size': [3, 64, 64],
        'class_dim': 25,
        'augment_path': work_dir+dataset_prefix+"_aug/", #增强后      ,
        'src_path': 'data/data55032/archive_train.zip',
        'target_path': work_dir+'/dataset/'+dataset_prefix + '/',
        'train_txt': work_dir+ dataset_prefix + '_train.txt',
        'eval_txt': work_dir+ dataset_prefix + '_eval.txt',
        'label_dict': {},
        'num_epochs': 10,
        'batch_size': 8,
        'learning_strategy': {'lr': 0.001},
        "img_enhance": True,
        'gen_img_count': 2,
    }
    train_data, eval_data = gen_train_eval(src_path=train_params['src_path'],
                                           target_path=train_params['target_path'],
                                           augment_path=train_params['augment_path'],
                                           train_txt=train_params['train_txt'],
                                           eval_txt=train_params['eval_txt'],
                                           img_enhance=train_params["img_enhance"],
                                           gen_img_count=train_params['gen_img_count'])

    dic = train_data[['lbl', 'gem_name']].drop_duplicates()
    train_params['label_dict'] = {v: int(k) for k, v in dic.to_records(index=False)}
    train_params['class_dim'] = len(dic)


    #train_data、eval_data、train_params保存到文件中
    train_data.to_json(train_df_json)
    eval_data.to_json(eval_df_json)
    with open(train_params_path, 'w') as f:
        tmp_ = json.dumps(train_params)
        f.write(tmp_)

    # 从文件中读
    train_data = pd.read_json(train_df_json)
    eval_data = pd.read_json(eval_df_json)
    with open(train_params_path, 'r') as f:
        tmp_str = f.read()
    train_params = json.loads(tmp_str)
    print(train_params)

    '''
    构造数据提供器
    '''
    import paddle
    train_reader = paddle.batch(data_reader(train_data),
                                batch_size=train_params['batch_size'],
                                drop_last=True)

    eval_reader = paddle.batch(data_reader(eval_data),
                               batch_size=train_params['batch_size'],
                               drop_last=True)

if __name__=='__main__':demo()
