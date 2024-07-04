#! /bin/bash
tasks=(assembling-kits-3dtoolkit)
n_rotations=180
n_demos=(1)
n_aligns=(18)
n_steps=10000
interval=2000
n_tests=(2000 4000 6000 8000 10000)
agent=so2-align
seed=0
gpu_id=0

for task in ${tasks[*]};
do      
    for n_demo in ${n_demos[*]};
    do  
        for n_align in ${n_aligns[*]};
        do
            echo Train Task: ${task}-${n_rotations}-${n_demo}-${seed}/${agent}h${n_align}bk --gpu_id ${gpu_id};
            gnome-terminal --wait -- /bin/bash  \
                -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                echo Training Task: ${task}-${n_rotations}-${n_demo}-${seed}/${agent}h${n_align};   \
                python train.py --task ${task} --n_demos ${n_demo} --n_rotations ${n_rotations} \
                    --agent ${agent} --n_align ${n_align} --n_steps ${n_steps} --interval ${interval} \
                    --gpu_id ${gpu_id} --logging --seed ${seed} --config_file train.yaml --postfix h${n_align}bk; \

                sleep 30;"
            sleep 1

            # for n_test in ${n_tests[*]};
            # do
            echo Test Task: ${task}-${n_rotations}-${n_demo}-${seed}/${agent}h${n_align}bk --gpu_id ${gpu_id};
            gnome-terminal --wait -- /bin/bash  \
                -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                echo Testing Task: ${task}-${n_rotations}-${n_demo}-${seed}/${agent}h${n_align} --gpu_id ${gpu_id};   \
                python gripper_test.py --task ${task} --n_demos ${n_demo} --n_rotations ${n_rotations} \
                    --agent ${agent} --n_align ${n_align} --gpu_id ${gpu_id} \
                    --seed ${seed} --entire --config_file train.yaml --postfix h${n_align}bk --disp # --n_steps ${n_tests}; \

                sleep 30;"
            sleep 1
            # done;
        done;
    done
done;
echo Done!
