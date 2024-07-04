#! /bin/bash
tasks=(assembling-single-toolkit)
n_demos=(100 10 1)
n_rotations=(180)
n_align=12
n_step=10000
interval=2000
n_tests=(2000 4000 6000 8000 10000)
agents=(so2-align)
seed=0
gpu_id=1

for task in ${tasks[*]};
do
    for n_demo in ${n_demos[*]};
    do
        for n_rotation in ${n_rotations[*]};
        do  
            for agent in ${agents[*]};
            do
                # training
                echo Train Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent} --gpu ${gpu_id} --postfix 1x1;
                gnome-terminal --wait -- /bin/bash  \
                    -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                    echo Training Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent};    \
                    python train.py --task ${task} --n_demos ${n_demo} --n_rotations ${n_rotation} \
                            --agent ${agent} --n_align ${n_align} --n_steps ${n_step} --interval ${interval}    \
                            --gpu_id ${gpu_id} --logging --seed ${seed} --postfix 1x1; \

                    sleep 30;"
                sleep 1

                # testing
                echo Test Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent} --gpu ${gpu_id} --postfix 1x1;
                gnome-terminal --wait -- /bin/bash  \
                    -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                    echo Testing Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent};    \
                    python gripper_test.py --task ${task} --n_demos ${n_demo} --n_rotations ${n_rotation}  \
                            --agent ${agent} --n_align ${n_align} --gpu_id ${gpu_id}    \
                            --seed ${seed} --disp  --entire --postfix 1x1 # --n_steps 4000;  \ 

                    sleep 30;"
                sleep 1
            done;
        done;
    done;
done;
echo Done!
echo Done!
