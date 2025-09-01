CUDA_VISIBLE_DEVICES=0,1 accelerate launch run_tuning_1.py --config config/car-tree-tune.yaml
CUDA_VISIBLE_DEVICES=0,1 accelerate launch test_fatezero.py --config config/shape/car-tree.yaml

