
export PYTHONHASHSEED=42

mkdir -p results

echo "Running standard (Large) Transformer..."
python src/main.py \
    --data_path "./data" \
    --batch_size 16 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
    --epochs 15 \
    --lr 0.0003 \
    --max_samples 5000 \
    --max_length 64 \
    --save_path "./results" \
    --seed 42

if [ $? -eq 0 ]; then
    echo "Standard (Large) Transformer training completed successfully."
    
    echo "Running Small Transformer (ablation study)..."
    python src/main.py \
        --data_path "./data" \
        --batch_size 16 \
        --d_model 256 \
        --nhead 4 \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --dim_feedforward 512 \
        --dropout 0.1 \
        --epochs 15 \
        --lr 0.0003 \
        --max_samples 5000 \
        --max_length 64 \
        --save_path "./results" \
        --seed 42 \
        --use_small_model 

    if [ $? -eq 0 ]; then
        echo "Ablation study (Small model) completed successfully."
        
        echo "Plotting size ablation comparison results..."
        python src/plot_results.py
        
        echo "Size ablation study and comparison plotting completed! Compare results in ./results/"
    else
        echo "Ablation study (Small model) failed."
    fi
else
    echo "Standard (Large) Transformer training failed."
fi