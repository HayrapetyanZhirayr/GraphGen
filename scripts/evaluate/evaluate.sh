python3 -m graphgen.evaluate --folder /workspace/hayrapetyan/GG_data/Qwen2.5-72B-IT-CFA/data/graphgen/1760391060 \
                    --output /workspace/hayrapetyan/GG_data/Qwen2.5-72B-IT-CFA/data/graphgen/1760391060 \
                    --reward "OpenAssistant/reward-model-deberta-v3-large-v2,BAAI/IndustryCorpus2_DataRater" \
                    --uni MingZhong/unieval-sum \
                    --tokenizer "Qwen/Qwen3-14B"
