set -x data_partition train1
set -x raw_data_dir {$HOME}/Research/datasets/argoverse-tracking/{$data_partition}/
set -x dets_dataroot {$HOME}/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/
set -x results_dir {$HOME}/Research/MOT-Research/pyDort/results/results_pcd
set -x conf_dir {$HOME}/Research/MOT-Research/pyDort/conf
set -x scripts_dir {$HOME}/Research/MOT-Research/pyDort/scripts/trackers
set -x exp_no 6
set -x n_logs 100

# resnet dgcnn1024 dgcnn2048
#cv-cv-v0 cv-ctrv
for pcd_model in pointnet dgcnn1024 dgcnn2048;
    for motion_model in cv-cv;
        set -x out_suffix exp{$exp_no}_{$motion_model}_pcd_{$pcd_model}_tracker
        set -x out_dir {$results_dir}/{$data_partition}/tracks_dump_{$out_suffix}
        mkdir -p $out_dir

        for obj_cls in VEHICLE PEDESTRIAN;
            python {$scripts_dir}/run_tracker_pcd_filterpy.py \
                --dets_dataroot $dets_dataroot \
                --raw_data_dir $raw_data_dir \
                --min_hits 1 --max_age 5 --tracks_dump_dir $out_dir \
                --config_file {$conf_dir}/conf-{$motion_model}.json \
                --target_cls $obj_cls --pcd_gpu --pcd_model $pcd_model \
                --pcd_chunk_size 10 --n_logs $n_logs > {$out_dir}/{$obj_cls}_out1_{$out_suffix}.txt

            # Evaluate
            python {$scripts_dir}/../eval_tracking.py \
                --path_tracker_output $out_dir \
                --path_dataset $raw_data_dir \
                --d_max 30 --category $obj_cls
        end
    end
end

# python scripts/run_tracker_contrastive_multi_modal_filterpy.py \
#     --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ \
#     --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ \
#     --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp3_mm_tracker \
#     --config_file conf/conf-cv-cv.json \
#     --target_cls PEDESTRIAN --im_gpu --im_model convnext_tiny --pcd_model dgcnn1024 \
#     --agr avg --im_chunk_size 5 --pcd_chunk_size 5 > results/out3.txt
