for dirname in "outputs/scores" "outputs/snapshots" "outputs/logs";do
    if [ ! -d $dirname ]; then
        mkdir -p $dirname
    fi
done

max_epoch=20
dataset=thumos14
epoch=19
for mod in flow rgb;do
    exp_name=${dataset}_${mod}
    
    cfg_path=data/cfgs/${exp_name}.yml
    python train_net.py $dataset  --cfg ${cfg_path} --snapshot_pref outputs/snapshots/${exp_name}/ --epochs ${max_epoch}

    python test_net.py $dataset outputs/snapshots/${exp_name}/${dataset}_epoch_${epoch}_checkpoint.pth.tar outputs/scores/${exp_name} --cfg ${cfg_path}
    # python eval.py $dataset outputs/scores/${exp_name} -j8 --cfg ${cfg_path}   --nms_threshold 0.4
done

python eval.py ${dataset} outputs/scores/${dataset}_rgb outputs/scores/${dataset}_flow --score_weights 1 1.2 --nms_thr 0.4 -j8 --cfg $cfg_path