


echo "## Run"
task_name="sonba"
learning_rate=3e-5
num_train_epochs=12
train_batch_size=8
bert_model="vinai/phobert-base-v2"
data_dir="tmp_data/vlsp_MoRe_PHO_kc_image/VLSP2018"
cache_dir="cache"
max_seq_length=256

python train_bert_crf_EC_new_roberta.py \
    --do_train \
    --do_eval \
    --output_dir "./VLSP2021_img" \
    --bert_model "${bert_model}" \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --task_name "${task_name}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}