set -x data_partition train1
set -x raw_data_dir {$HOME}/Research/datasets/argoverse-tracking/{$data_partition}/
set -x dets_dataroot {$HOME}/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/
set -x results_dir {$HOME}/Research/MOT-Research/pyDort/results/results_simple
set -x conf_dir {$HOME}/Research/MOT-Research/pyDort/conf
set -x scripts_dir {$HOME}/Research/MOT-Research/pyDort/scripts/trackers
set -x exp_no 1
set -x n_logs 100

# cv-cv-v0 cv-ctrv
for motion_model in cv-cv;
    set -x out_suffix exp{$exp_no}_{$motion_model}_simple_tracker
    set -x out_dir {$results_dir}/{$data_partition}/tracks_dump_{$out_suffix}
    mkdir -p $out_dir

    for obj_cls in VEHICLE PEDESTRIAN;
        # Run tracker
        python {$scripts_dir}/run_tracker_filterpy.py \
            --dets_dataroot $dets_dataroot \
            --raw_data_dir $raw_data_dir \
            --min_hits 1 --max_age 5 --tracks_dump_dir $out_dir \
            --config_file {$conf_dir}/conf-{$motion_model}.json \
            --target_cls $obj_cls --n_logs $n_logs > {$out_dir}/{$obj_cls}_out1_{$out_suffix}.txt

        # Evaluate
        python {$scripts_dir}/../eval_tracking.py \
            --path_tracker_output $out_dir \
            --path_dataset $raw_data_dir \
            --d_max 30 --category $obj_cls
    end
end
