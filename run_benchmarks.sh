# python run_benchmark.py dynamo --speedup-parameterized-cudagraphs --inductor --performance --inference --huggingface 2>&1 | tee huggingface.log

# python run_benchmark.py dynamo --speedup-parameterized-cudagraphs --performance --inference --only=Speech2Text2ForCausalLM # 2>&1 | tee huggingface.log

python run_benchmark.py dynamo --speedup-parameterized-cudagraphs --performance --inference --huggingface # 2>&1 | tee huggingface.log

python run_benchmark.py dynamo --speedup-parameterized-cudagraphs --performance --inference --torchbench # 2>&1 | tee huggingface.log

# Timm models fail wit this for some reason:
# FileNotFoundError: [Errno 2] No such file or directory: '/home/dgalvez/code/asr/pytorch-6/galvez_benchmarks/benchmark/userbenchmark/dynamo/dynamobench/timm_models.yaml'
python run_benchmark.py dynamo --speedup-parameterized-cudagraphs --performance --inference --timm # 2>&1 | tee huggingface.log
