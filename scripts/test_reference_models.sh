# test.sh
if [ ! -d "outputs/scores" ]; then
    mkdir -p outputs/scores
fi

dataset=$1

if [[ $dataset == "thumos14" ]];then
    python test_net.py thumos14 reference_models/thumos14_flow.pth.tar outputs/scores/thumos14_flow --cfg data/cfgs/thumos14_flow.yml
    python eval.py thumos14 outputs/scores/thumos14_flow -j8 --cfg data/cfgs/thumos14_flow.yml --nms_threshold 0.4|tee -a outputs/eval.log

    python test_net.py thumos14 reference_models/thumos14_rgb.pth.tar outputs/scores/thumos14_rgb --cfg data/cfgs/thumos14_rgb.yml
    python eval.py thumos14 outputs/scores/thumos14_rgb -j8 --cfg data/cfgs/thumos14_rgb.yml --nms_threshold 0.4|tee -a outputs/eval.log

    # FUSE
    python eval.py thumos14 outputs/scores/thumos14_rgb outputs/scores/thumos14_flow --score_weights 1 1.2 --nms_thr 0.4 -j8 --cfg data/cfgs/thumos14_rgb.yml |tee -a outputs/eval.log

elif [[ $dataset == "muses" ]]; then
    python test_net.py muses reference_models/muses.pth.tar outputs/scores/muses --cfg data/cfgs/muses.yml

    python eval.py muses outputs/scores/muses --nms_thr 0.4 -j8 --cfg data/cfgs/muses.yml |tee -a outputs/muses_eval.log

else
    echo "unsupported dataset ${dataset}"
fi