#! /bin/bash
task_ids=(0)
tasks=(align-box-corner)
n_demos=(10)
n_rotations=(36)
n_aligns=(2 4 6 12 18 36)
n_steps=10000
interval=2000
# n_tests=(2000 4000 6000 8000 10000)
agent=so2-align
seed=0
gpu_id=1

# pwd
for i in ${task_ids[*]};
do      
    for n_rotation in ${n_rotations[*]};
    do  
        for n_align in ${n_aligns[*]};
        do
        echo Task: ${tasks[${i}]}-${n_rotation}-${n_demos[${i}]}-${seed}/${agent}h${n_align};
        python train.py --task ${tasks[${i}]} --n_demos ${n_demos[${i}]} --n_rotations ${n_rotation} \
                --agent ${agent} --n_align ${n_align} --n_steps ${n_steps} --interval ${interval} \
                --gpu_id ${gpu_id} --logging --postfix h${n_align} --seed ${seed}
        python gripper_test.py --task ${tasks[${i}]} --n_demos ${n_demos[${i}]} --n_rotations ${n_rotation} \
                --agent ${agent} --n_align ${n_align} --n_steps ${n_steps} --gpu_id ${gpu_id} \
                --disp --postfix h${n_align} --entire --seed ${seed}
        done;
    done
done;
