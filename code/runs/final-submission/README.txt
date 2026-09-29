MP1 final submission: depth-10 hybrid (neural + ngram order 2-10 + cache + copy), EMA weights at step 1000.
Use: .venv/Scripts/python.exe evaluate.py --checkpoint runs/final-submission/checkpoint.pt --split test --device cpu --precision fp32 --threads 3
Python 3.12, torch 2.7.1, CPU FP32.
