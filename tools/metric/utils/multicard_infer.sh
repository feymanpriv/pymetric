#config_path=$1

rm -rf ./eval_outputs/combine_results/*
total_num=12
for cutno in $(seq 1 $total_num);
do
    {
    let gpu_id=cutno%4
    export CUDA_VISIBLE_DEVICES=$gpu_id
    cmd="python tools/metric/infer.py --cfg configs/metric/resnst269_8gpu.yaml INFER.TOTAL_NUM ${total_num} INFER.CUT_NUM ${cutno}"
    echo [start cmd:] ${cmd}
    echo ${cmd} | sh
    } &
    sleep 0.5
done
wait
echo "run_cal_logo_fea.sh finished~"

