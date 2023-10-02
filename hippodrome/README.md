

## instances

```bash
wget https://cloudtcs.tcs.uni-luebeck.de/index.php/s/Dm5pZfzxkoP8cL6/download/exact-public.zip
wget https://cloudtcs.tcs.uni-luebeck.de/index.php/s/eitXacoY5ACHqtx/download/exact-private.zip
```

```bash
mkdir -p instances/pace2023
unzip exact-public.zip -d instances/pace2023
unzip exact-private.zip -d instances/pace2023
mv instances/pace2023/exact-public/*.gr.xz instances/pace2023
mv instances/pace2023/exact-private/*.gr.xz instances/pace2023
rm -r instances/pace2023/exact-public
rm -r instances/pace2023/exact-private
rm -r instances/pace2023/__MACOSX
```

```bash
for file in $( ls instances/pace2023/*.gr.xz )
do
  [[ -f "${file%.xz}" ]] || xz --decompress ${file}
done

```