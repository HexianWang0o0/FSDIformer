num_process=8
master_port=12345
ip=The_ip_address_of_the_machine
machines=1
mach_rank=0
precision=no

model_name=FSDIformer
root_path_name=./datasets/
data_path_name=windfarm1.csv
task_name=WPF
seq_len=96
pred_len=4
batch_size=32
checkpoints_path=./your_path_to_save_model

accelerate launch --multi_gpu --mixed_precision $precision --num_processes $num_process\
  --num_machines $machines --machine_rank $mach_rank --main_process_port $master_port\
  --main_process_ip $ip run.py \
  --task_name $task_name\
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 3 \
  --e_layers 3 \
  --n_heads 8 \
  --dropout 0.1\
  --des 'Exp' \
  --train_epochs 30\
  --patience 8\
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate 0.001\
  --checkpoints $checkpoints_path \
  --selected_freq_count 32\
  --levels 3\
