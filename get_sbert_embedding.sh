MODEL=$1
K=16

python get_sbert_embedding.py --sbert_model $MODEL --task MELD
python get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task MELD

for seed in 13 21 87 100
do
    for task in MELD
    do
        cp data/k-shot/$task/$K-42/test_sbert-$MODEL.npy  data/k-shot/$task/$K-$seed/
    done

done
