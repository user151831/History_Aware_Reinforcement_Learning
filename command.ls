conda create --name RL_env --file requirements_training.txt
conda activate RL_env

for dataset in SimBank bpic2012_accepted bpic2017_accepted hospital_billing_2; do
    for order in {1..10}; do
        for run_id in 0 1 2 3 4; do
            python -u src/CQL_offline.py \
                --dataset $dataset \
                --method CQL \
                --batch_size 100 \
                --target_update 1000 \
                --n_hidden1 300 \
                --n_hidden2 400 \
                --deep_network True \
                --num_steps 2000000 \
                --num_steps_ope 600000 \
                --order $order \
                --run_id $run_id
        done
    done
done

for dataset in sepsis_cases_1 sepsis_cases_2; do
    for order in {1..10}; do
        for run_id in 0 1 2 3 4; do
            python -u src/CQL_offline.py \
                --dataset $dataset \
                --method CQL \
                --batch_size 100 \
                --target_update 1000 \
                --n_hidden1 300 \
                --n_hidden2 400 \
                --deep_network True \
                --num_steps 2000000 \
                --num_steps_ope 3000000 \
                --order $order \
                --run_id $run_id
        done
    done
done


for dataset in SimBank bpic2012_accepted bpic2017_accepted hospital_billing_2; do
    for order in {1..10}; do
        for run_id in 0 1 2 3 4; do
            python -u src/CQL_offline.py \
                --dataset $dataset \
                --method DoubleDQN \
                --batch_size 100 \
                --target_update 1000 \
                --n_hidden1 300 \
                --n_hidden2 400 \
                --deep_network True \
                --num_steps 2000000 \
                --num_steps_ope 600000 \
                --order $order \
                --run_id $run_id
        done
    done
done

for dataset in sepsis_cases_1 sepsis_cases_2; do
    for order in {1..10}; do
        for run_id in 0 1 2 3 4; do
            python -u src/CQL_offline.py \
                --dataset $dataset \
                --method DoubleDQN \
                --batch_size 100 \
                --target_update 1000 \
                --n_hidden1 300 \
                --n_hidden2 400 \
                --deep_network True \
                --num_steps 2000000 \
                --num_steps_ope 3000000 \
                --order $order \
                --run_id $run_id
        done
    done
done

for dataset in SimBank bpic2012_accepted bpic2017_accepted hospital_billing_2 sepsis_cases_1 sepsis_cases_2; do
    for run_id in 0 1 2 3 4; do
        python -u src/Baseline_online_DQN.py \
            --dataset $dataset \
            --method Online_DoubleDQN \
            --batch_size 100 \
            --target_update 1000 \
            --n_hidden1 300 \
            --n_hidden2 400 \
            --deep_network True \
            --num_steps 2000000 \
            --num_steps_ope 600000 \
            --order 1 \
            --run_id $run_id
    done
done

conda create --name MA_test_env --file requirements.txt
conda activate MA_test_env

python -u src/MA_test.py --T 12 --n_cases 1000 --max_lag 7 --L 3 --Q 4 --dataset SimBank --n_samples 100 --MDP_model HMDP
python -u src/MA_test.py --n_cases 1000 --max_lag 10 --L 3 --Q 8 --dataset bpic2012_accepted --n_samples 100
python -u src/MA_test.py --n_cases 500 --max_lag 10 --L 3 --Q 8 --dataset bpic2017_accepted --n_samples 100
python -u src/MA_test.py --n_cases 1000 --max_lag 4 --L 3 --Q 3 --dataset hospital_billing_2 --n_samples 100
python -u src/MA_test.py --n_cases 200 --max_lag 7 --L 3 --Q 4 --dataset sepsis_cases_1 --n_samples 100
python -u src/MA_test.py --n_cases 200 --max_lag 7 --L 3 --Q 4 --dataset sepsis_cases_2 --n_samples 100
