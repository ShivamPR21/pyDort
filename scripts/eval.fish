set -x scripts_dir {$HOME}/Research/MOT-Research/pyDort/scripts/
set -x path_tracker_output {$HOME}/Research/MOT-Research/pyDort/results_/train1/tracks_dump_exp1_cv-cv_simple_tracker
set -x path_dataset $HOME/Research/datasets/argoverse-tracking/train1/

for category in VEHICLE PEDESTRIAN;
    for d_max in 100 50 30;
        python {$scripts_dir}/eval_tracking.py --path_tracker_output $path_tracker_output \
            --path_dataset $path_dataset --d_max $d_max --category $category
    end
end
