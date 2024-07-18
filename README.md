# DAC-pytorch

Deep Adaptive Image Clustering pytorch version

You should run this code in cuda environment

## Usage

### Train model and Inference

```bash
python3 main.py --batch_size 128
```

### Only Inference
If you want to only inference, you can load trained model(model_weights.pth).

```bash
python3 main.py --batch_size 128 --save_model_path './model_weights.pth' --load_model_path './model_weights.pth'
```

## Reference

https://github.com/GuHongyang/DAC-pytorch

