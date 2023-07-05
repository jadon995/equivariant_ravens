#! /bin/bash
task_ids=(0)
tasks=(stack-block-pyramid assembling-kits-3dtoolkit)
n_rotations=(36 360)
n_demos=(10)
n_aligns=(2 4 6 12 18 36)
n_steps=10000
interval=2000
n_tests=(2000 4000 6000 8000 10000)
agent=so2-align
seed=0
gpu_id=0

for i in ${task_ids[*]};
do      
    for n_demo in ${n_demos[*]};
    do  
        for n_align in ${n_aligns[*]};
        do
            echo Train Task: ${tasks[${i}]}-${n_rotations[${i}]}-${n_demo}-${seed}/${agent}h${n_align} --gpu_id ${gpu_id};
            gnome-terminal --wait -- /bin/bash  \
                -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                echo Training Task: ${tasks[${i}]}-${n_rotations[${i}]}-${n_demo}-${seed}/${agent}h${n_align};   \
                python train.py --task ${tasks[${i}]} --n_demos ${n_demo} --n_rotations ${n_rotations[${i}]} \
                    --agent ${agent} --n_align ${n_align} --n_steps ${n_steps} --interval ${interval} \
                    --gpu_id ${gpu_id} --logging --postfix h${n_align} --seed ${seed}; \

                sleep 30;"
            sleep 1

            # for n_test in ${n_tests[*]};
            # do
            echo Test Task: ${tasks[${i}]}-${n_rotations[${i}]}-${n_demo}-${seed}/${agent}h${n_align} --gpu_id ${gpu_id};
            gnome-terminal --wait -- /bin/bash  \
                -c "source ~/.bashrc; cd ~/workspace/equivariant_ravens; pwd;   \
                echo Testing Task: ${tasks[${i}]}-${n_rotations[${i}]}-${n_demo}-${seed}/${agent}h${n_align} --gpu_id ${gpu_id};   \
                python gripper_test.py --task ${tasks[${i}]} --n_demos ${n_demo} --n_rotations ${n_rotations[${i}]} \
                    --agent ${agent} --n_align ${n_align} --gpu_id ${gpu_id} \
                    --postfix h${n_align} --seed ${seed} --entire #--disp --n_steps ${n_tests}; \

                sleep 30;"
            sleep 1
            # done;
        done;
    done
done;
echo Done!
