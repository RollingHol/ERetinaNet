import os
import xml.etree.ElementTree as ET

from utils.utils import get_classes

# Adjust the VOC dataset format
def convert_annotation(VOCdevkit_path, classes_path, year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    classes, _ = get_classes(classes_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

def datasetFormat(data_dir, classes_path):
    print("Generate train.txt, val.txt and test.txt.")
    year = str(os.listdir(data_dir)[0])[3:]
    VOCdevkit_sets = [(year, 'train'), (year, 'val'), (year, 'test')]
    train_txt = year+'_train.txt'
    val_txt = year+'_val.txt'
    test_txt = year+'_test.txt'
    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(data_dir, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(data_dir), year, image_id))
            convert_annotation(data_dir, classes_path, year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("Generate train.txt, val.txt and test.txt done.")
    return train_txt, val_txt, test_txt
