# Evaluation


Create python environment and install dependencies.
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make test data available.
```shell
python3 scripts/pipeline.py download convert generate
```

Run experiments (needs the test data).
```shell
python3 scripts/pipeline.py md
```

Run all algorithms on all available data.
```shell
cargo run --bin evaluation --release data/02-graphs
```

Execute one of the algorithms on a single file.
```shell
cargo run --bin md --release -- --input-type metis --input <INPUT> --output <OUTPUT> --algo kar19-rust
```
