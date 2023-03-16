META_PATH="/finetune/v1.1/squad_full_meta_data"
export PYTHONPATH=$PYTHONPATH:/workspace/models
DEFAULTSKIP=""
DEFAULTMODE="squad"
MODE=${1:-$DEFAULTMODE}
SKIP=${2:-$DEFAULTSKIP}
if [[ -z "$SKIP" ]]
then
  echo skip
else
 pip3 install tensorflow_hub latest
 pip install gin-config
 pip install -r ../../requirements.txt
fi

export MODEL_BERT_GOOGLE_TF1_CKPT_GOOGLE_INTERNAL=/mnt/nvdl/datasets/joc-datasets/model/bert_google_tf1/ckpt/google_internal # key: model/bert_google_tf1/ckpt/google_internal

export BERT_DIR=/pretrain

# For dumping HLOs
# /tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/generated --xla_gpu_enable_cublaslt=true" python $SCRIPT \
if [[ $MODE == "squad" ]]; then
  TF_CPP_VMODULE=gemm_rewriter=1 TF_XLA_FLAGS="--tf_xla_auto_jit=2"  XLA_FLAGS="--xla_gpu_enable_cublaslt=true" \
  python run_squad.py \
    --input_meta_data_path=/finetune/v1.1/squad_full_meta_data \
    --train_data_path=/finetune/v1.1/data/squad_train.tf_record \
    --predict_file=/finetune/v1.1/dev-v1.1.json \
    --vocab_file=${BERT_DIR}/vocab.txt \
    --bert_config_file=${BERT_DIR}/bert_config.json \
    --train_batch_size=4 \
    --predict_batch_size=4 \
    --mode=train_and_predict \
    --learning_rate=8e-5 \
    --num_train_epochs=2 \
    --model_dir=/joc/my_bert_model \
    --enable_xla=true \
    --dtype=fp16 \
    --distribution_strategy=mirrored
elif [[ $MODE == "pretrain" ]]; then
  TF_CPP_VMODULE=gemm_rewriter=1 TF_XLA_FLAGS="--tf_xla_auto_jit=2"  XLA_FLAGS="--xla_gpu_enable_cublaslt=true" \
  python run_pretraining.py \
    --input_files=/tfrecord/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training/wiki_books_corpus_training_*.tfrecord \
    --num_train_epochs=10 \
    --bert_config_file=${BERT_DIR}/bert_config.json \
    --train_batch_size=4 \
    --enable_xla=false \
    --dtype=fp16 \
    --distribution_strategy=mirrored
else
  echo Invalid mode
fi
