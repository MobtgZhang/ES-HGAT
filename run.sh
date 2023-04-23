model=("ESHGAT1" "ESHGAT2" "ESHGAT" "GraphESHGAT" "ESHGAT4" "ESHGAT5" "ESCapsHGAT" "HyperGATPre" "PretrainingModel" "TextCNNPre" "TextGCNPre" "CapsESHGAT" "CapsuleNetPre")
batch_size=64
epoches=4
learning_rate=5e-5
english_pretrain_list=("xlnet-base-cased" "bert-base-uncased" )

for tp_model in ${model[@]}
do
    for english_pretrain in ${english_pretrain_list[@]}
    do
        rm -rf ./result
        python build.py --dataset N15News
        python main.py --dataset N15News --model-name $tp_model --batch-size $batch_size  --epoches $epoches --learning-rate $learning_rate --pretrain-path $english_pretrain
    done
done
