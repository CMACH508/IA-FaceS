from deepface import DeepFace
import argparse
import os
from tqdm import tqdm

dicts = {"mouth_open": "mouth_close", "mouth_close": "mouth_open",
         "young": "old", "beard": "no_beard", "bushy": "arched", "arched": "bushy",
         "narrow_eyes": "no_narrow_eyes", "male": "female", "female": "male",
         "smile": "no_smile", "bags": "no_bags", "big_nose": "pointy_nose", "big_lips": "no_big_lips"}

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths


def evaluate(args):
    img_list = make_dataset_txt('data/attr_lists/%s.txt' % dicts[args.attr])

    j = 0
    total = 0

    note_detected = make_dataset_txt('data/not_detecetd/%s.txt' % args.method)

    for i, name in enumerate(img_list):
        if name in note_detected:
            img_list.pop(i)

    print("Total faces:", len(img_list))

    for name in tqdm(img_list):
        name = name.split('.')[0]
        # print(os.path.join(args.fake, '%s' % args.attr, f"{name}.png"))
        try:
            result = DeepFace.verify(os.path.join(args.real, f"{name}.png"),
                                     os.path.join(args.fake, f"{name}.png"),
                                     model_name=models[6])
            total += result['distance']

        except:
            j += 1

    print("Not detected:", j)
    print("Similarity on detected faces:", total / (len(img_list) - j))


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=str, default=None)
    parser.add_argument("--real", type=str, default=None)
    parser.add_argument("--attr", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()
    evaluate(args)
