dataset=("CHIP-CTC" "CLUEEmotion2020" "IndustryData" "N15News")
model=("CapsESHGAT" "CapsuleNetPre" "ESHGAT" "GraphESHGAT" "HyperGATPre" "PretrainingModel" "TextCNNPre" "TextGCNPre")
batch_size=12
epoches=20

for tp_dataset in ${dataset[@]}
do
    for tp_model in ${model[@]}
    do
        python main.py --dataset $tp_dataset --model-name $tp_model --batch-size $batch_size  --epoches $epoches
    done
done
