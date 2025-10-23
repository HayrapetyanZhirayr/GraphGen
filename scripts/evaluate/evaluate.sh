python3 -m graphgen.evaluate \
                    --input "/workspace/hayrapetyan/GG_data/EDA/tmp_data/GG_fin_loc_iqs.jsonl" \
                    --output "/workspace/hayrapetyan/GG_data/EDA/tmp_data/GG_fin_loc_iqs_gg.jsonl" \
                    --reward "OpenAssistant/reward-model-deberta-v3-large-v2,BAAI/IndustryCorpus2_DataRater" \
                    --uni MingZhong/unieval-sum \
