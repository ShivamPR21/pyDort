set -x data_partition train1
set -x raw_data_dir {$HOME}/Research/datasets/argoverse-tracking/{$data_partition}/
set -x dets_dataroot {$HOME}/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/
set -x results_dir {$HOME}/Research/MOT-Research/pyDort/results
set -x conf_dir {$HOME}/Research/MOT-Research/pyDort/conf
set -x scripts_dir {$HOME}/Research/MOT-Research/pyDort/scripts/trackers
set -x exp_no 4

# resnet18 resnet50
for model in resnet18 resnet50;
# cv-cv-v0 cv-ctrv
    for motion_model in cv-cv;
        set -x out_suffix exp{$exp_no}_{$motion_model}_contrastive_{$model}_tracker
        set -x out_dir {$results_dir}/{$data_partition}/tracks_dump_{$out_suffix}
        mkdir -p $out_dir

        for obj_cls in VEHICLE PEDESTRIAN;
            python {$scripts_dir}/run_tracker_contrastive_filterpy.py \
                --dets_dataroot $dets_dataroot \
                --raw_data_dir $raw_data_dir \
                --min_hits 1 --max_age 5 --tracks_dump_dir $out_dir \
                --config_file {$conf_dir}/conf-{$motion_model}.json \
                --target_cls $obj_cls --gpu --model $model --agr avg --chunk_size 2 > {$out_dir}/{$obj_cls}_out1_{$out_suffix}.txt
        end
    end
end

# python scripts/run_tracker_contrastive_filterpy.py \
#     --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ \
#     --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ \
#     --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp3_resnet18_cl_tracker \
#     --config_file conf/conf-cv-cv.json \
#     --target_cls PEDESTRIAN --gpu --model resnet18 --agr avg --chunk_size 2 > results/out4.txt
