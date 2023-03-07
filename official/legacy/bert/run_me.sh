META_PATH="/finetune/v1.1/squad_full_meta_data"
export PYTHONPATH=$PYTHONPATH:/workspace/models
pip3 install tensorflow_hub latest
pip install gin-config
pip install -r ../../requirements.txt
export MODEL_BERT_GOOGLE_TF1_CKPT_GOOGLE_INTERNAL=/mnt/nvdl/datasets/joc-datasets/model/bert_google_tf1/ckpt/google_internal # key: model/bert_google_tf1/ckpt/google_internal

export BERT_DIR=/pretrain
DEFAULTSCRIPT="run_squad.py"
SCRIPT=${1:-$DEFAULTSCRIPT}
echo $SCRIPT

TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/generated --xla_gpu_enable_cublaslt=true" python $SCRIPT \
 --input_meta_data_path=${META_PATH} \
 --train_data_path=/finetune/v1.1/data/squad_train.tf_record \
 --predict_file=/finetune/v1.1/dev-v1.1.json \
 --vocab_file=${BERT_DIR}/vocab.txt \
 --bert_config_file=${BERT_DIR}/bert_config.json \
 --train_batch_size=4 \
 --predict_batch_size=4 \
 --learning_rate=8e-5 \
 --num_train_epochs=2 \
 --model_dir=/joc/my_bert_model \
 --enable_xla=true \
 --dtype=fp16 \
 --distribution_strategy=mirrored
       &> log.tmp


