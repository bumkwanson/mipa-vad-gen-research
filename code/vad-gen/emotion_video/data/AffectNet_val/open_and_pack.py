from PIL import Image
import numpy as np
import pathlib
import os

def main():
    file_dir = pathlib.Path(__file__).parent.resolve()
    images = []
    labels = []
    img_paths = [
        "Anger/006dbcccdcd992be19ab3a5751c24bcaf50ecb33d8ec781ae6d3f5c0.png",
        "Disgust/007ff33d095624d57968749d92da0d40cecff2e93904ac9224c711b1.jpg",
        "Fear/0026a04d7f5c60a645979f715c9dccba75c043a98eb3be843dc75bde.jpeg",
        "Happiness/00ff0089ad7ca210b0bd2fe2eba7ae1312fcc387846588872add238d.jpg",
        "Neutral/0147ed1ae450871793854d373a01241581e93918ad0871b3497d457f.JPG",
        "Sadness/00411982cad24a500062d3b8a5dd441a79c7bb6bdcef08bf78c007d3.jpg",
        "Surprise/00a11b6818d619356577c088ff6027651da6c90ff7a825cbf03e3395.JPG",
    ]
    for p in img_paths:
        img_path = os.path.join(file_dir, p)
        img_arr = np.array(Image.open(img_path))
        label = p.split('/')[0]
        labels.append(label)
        images.append(img_arr)
    #for label in os.listdir(file_dir):
    #    path = os.path.join(file_dir, label)
    #    if label.startswith('.') or not os.path.isdir(path):
    #        continue
    #    print("Processing {}...".format(label))
    #    for img in os.listdir(path):
    #        if img.startswith('.'):
    #            continue
    #        img_path = os.path.join(path, img)
    #        img_arr = np.array(Image.open(img_path))
    #        labels.append(label)
    #        images.append(img_arr)
    print("Almost done...")
    images = np.array(images)
    labels = np.array(labels)
    print(images.shape)
    print(labels.shape)
    images_name = 'AffectNet_val_imgs_7.npy'
    labels_name = 'AffectNet_val_lbls_7.npy'
    np.save(images_name, images)
    np.save(labels_name, labels)
    print("Done!")

if __name__ == '__main__':
    main()
