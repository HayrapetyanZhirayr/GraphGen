mkdir -p cache/atomic

for i in {1..25}; do
    mkdir -p cache/atomic/chunk_${i}
    python3 -m graphgen.generate \
        --config_file graphgen/configs/atomic_chunked/atomic_config_${i}.yaml \
        --output_dir cache/atomic/chunk_${i}
done