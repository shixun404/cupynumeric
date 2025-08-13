#!/home/shixunw/bin/zsh
mpirun -np 2 --report-bindings -x UCX_NET_DEVICES=mlx5_0:1 -mca pml ucx -mca coll ^ucc,hcoll \
    /home/shixunw/cupynumeric.internal/test_all2all.py > rank_${OMPI_COMM_WORLD_RANK}.out