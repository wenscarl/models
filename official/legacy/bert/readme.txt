# Testing Fp8 on SQUAD, a fine-tuning task

## Run SQUAD
bash run_me.sh run_squad.py

## Run test_einsum.py or test_encoder.py
bash run_me.sh test_einsum.py[test_encoder.py]

It's expected to see cublas\$lt\$matmul\f8 in the files located at /tmp/generated/