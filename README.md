# Modular decomposition


`miz23-md-cpp-sys` is a ffi wrapper for the modular partition part of the [twinwidth-2023](https://github.com/mogproject/twinwidth-2023) repository. To get the required cpp files and to apply a patch file, run the following command:
```shell
bash algorithms/miz23-md-cpp-sys/init.sh
```



```shell
cargo test --lib miz23-md-rs
cargo test --lib miz23-md-cpp
cargo bench --lib hippodrome
cargo run --bin hippodrome
cargo run --bin playground
```

```shell
python3 notes/jsc72.py
python3 notes/hm79.py
```