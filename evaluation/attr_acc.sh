FAKE=/path/to/attr manipulation

list="0 1 4 6 16 24"
for i in $list;
do
python attr_accuracy.py --fake $FAKE --attr_idx $i;
done

