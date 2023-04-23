dataset=("CHIP-CTC" "IndustryData" "CLUEEmotion2020")
model=("ESHGAT1" "ESHGAT2" "ESHGAT" "GraphESHGAT" "ESHGAT4" "ESHGAT5" "ESCapsHGAT" "HyperGATPre" "PretrainingModel" "TextCNNPre" "TextGCNPre" "CapsESHGAT" "CapsuleNetPre")
batch_size=32
epoches=10
learning_rate=5e-5
pretrain_path_all_lists=( "bert-base-chinese" "hfl/chinese-xlnet-base" )
for tp_dataset in ${dataset[@]}
do
    for tp_model in ${model[@]}
    do
        for pretrain_path in ${pretrain_path_all_lists[@]}
        do  
            rm -rf ./result
            python build.py --dataset $tp_dataset
            python main.py --dataset $tp_dataset --model-name $tp_model --batch-size $batch_size  --epoches $epoches --learning-rate $learning_rate --pretrain-path $pretrain_path
        done
    done
done
