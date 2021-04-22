#cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir="/home/caffe"

cd $root_dir

redo=1
data_root_dir="$HOME/data/train"
dataset_name="Custom"
mapfile="$HOME/data/labelmap.prototxt"
anno_type="detection"
db="lmdb"
min_dim=200
max_dim=1000
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python3 $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/FaceDetection/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
#  echo $anno_type $mapfile $min_dim $max_dim $width $height $extra_cmd $data_root_dir $root_dir/FaceDetection/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name

done
