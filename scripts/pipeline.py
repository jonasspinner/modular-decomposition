import run

exact_instance_names = [f"exact_{i:03}" for i in range(1, 201)]
run.add("exact",
        "target/release/md --input-type pace2023 --algo miz23-rust --input hippodrome/instances/pace2023/[[input]].gr --output [[output]]",
        {"input": exact_instance_names, "output": "hippodrome/instances/pace2023/[[input]].md"},
        creates_file="[[output]]")

heuristic_instance_names = [f"heuristic_{i:03}" for i in range(1, 201)]
run.add("heuristic",
        "target/release/md --input-type pace2023 --algo miz23-rust --input hippodrome/instances/pace2023/[[input]].gr --output [[output]]",
        {"input": heuristic_instance_names, "output": "hippodrome/instances/pace2023/[[input]].md"},
        creates_file="[[output]]")

run.run()
