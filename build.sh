dataset=("CHIP-CTC" "CLUEEmotion2020" "IndustryData" "N15News")

for tp_name in ${dataset[@]}
do
    python build.py --dataset $tp_name
    echo "Process dataset $tp_name"
done
