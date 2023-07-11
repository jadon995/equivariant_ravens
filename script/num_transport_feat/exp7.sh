#! /bin/bash
tasks=(assembling-kits-3dtoolkit)
n_demos=(100)
n_rotations=(360)
n_align=4
n_feat=3
n_step=10000
interval=2000
n_tests=(2000 4000 6000 8000 10000)
agents=(so2-align)
seed=0
gpu_id=0

for task in ${tasks[*]};
do
    for n_demo in ${n_demos[*]};
    do
        for n_rotation in ${n_rotations[*]};
        do  
            for agent in ${agents[*]};
            do
                # training
                echo Train Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent} --gpu ${gpu_id};
                gnome-terminal --wait -- /bin/bash  \
                    -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                    echo Training Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent};    \
                    python train.py --task ${task} --n_demos ${n_demo} --n_rotations ${n_rotation} \
                            --agent ${agent} --n_align ${n_align} --n_feat ${n_feat} --n_steps ${n_step} --interval ${interval}    \
                            --gpu_id ${gpu_id} --logging --seed ${seed} --postfix feat3; \

                    sleep 30;"
                sleep 1

                # testing
                echo Test Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent} --gpu ${gpu_id};
                gnome-terminal --wait -- /bin/bash  \
                    -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                    echo Testing Task: ${task}-${n_rotation}-${n_demo}-${seed}/${agent};    \
                    python gripper_test.py --task ${task} --n_demos ${n_demo} --n_rotations ${n_rotation}  \
                            --agent ${agent} --n_align ${n_align} --n_feat ${n_feat} --gpu_id ${gpu_id}    \
                            --seed ${seed} --entire --postfix feat3 #--disp  --n_steps ${n_tests};  \ 

                    sleep 30;"
                sleep 1
            done;
        done;
    done;
done;
echo Done!
