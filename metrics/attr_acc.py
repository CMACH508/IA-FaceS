import argparse
import os
import tensorflow as tf
from torchvision import transforms
from PIL import Image
from utils.classifier import make_classifier

import numpy as np

id_to_name = {
    0: ["male", "female"],
    1: ["smile", "no_smile"],
    4: ["young", "old"],
    6: ["arched", "bushy"],
    7: ["bags", "no_bags"],
    10: ["big_lips", "no_big_lips"],
    11: ["big_nose", "no_big_nose"],
    16: ["bushy", "arched"],
    22: ["makeup", "no_makeup"],
    24: ["mouth_open", "mouth_close"],
    25: ["mustache", "no_mustache"],
    26: ["narrow_eyes", "no_narrow_eyes"],
    27: ["no_beard", "beard"],
    30: ["pointy_nose", "no_pointy_nose"]
}

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def evaluate(attrib_idx):
    img_list = os.listdir(os.path.join(args.fake, id_to_name[attrib_idx][0]))
    classifier = make_classifier(attrib_idx)
    all_acc = 0
    for i in range(len(img_list) // 100):
        tem_img_list = img_list[i * 100:(i + 1) * 100]
        images = []
        for name in tem_img_list[:100]:
            fake_filename = os.path.join(args.fake, id_to_name[attrib_idx][0], name)
            fake_img = transform(Image.open(fake_filename).convert("RGB"))
            images.append(fake_img.numpy())
        images = np.stack(images, axis=0)
        logits = classifier.get_output_for(images, None)
        predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1)).eval()
        acc = np.sum((np.argmax(predictions, axis=1) == 1))
        all_acc += acc
    print(all_acc)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'acc.txt'), 'a') as f:
        f.write('%d' % all_acc)
        f.write('\n')


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default="output/", type=str)
    parser.add_argument("--fake", type=str, default=None)
    parser.add_argument("--attr_idx", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.attr_idx)
