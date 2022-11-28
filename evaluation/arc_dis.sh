FAKE=/path/to/edited_images
REAL=/path/to/reconstruction

python arc_dis.py --fake $FAKE --attr mouth_open --real $REAL --method iafaces;

