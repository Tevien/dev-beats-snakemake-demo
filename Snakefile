# this is my snakefile
configfile: "config.json"

rule all:
  input:
    "my_mnist.pkl",
    "my_mnist_model.h5",
    "test.png"

rule prep_inputs:
  output:
    "my_mnist.pkl"
  shell:
    "python prep_data.py --size 5000 --output {output}"

rule train:
  input:
    "my_mnist.pkl"
  output:
    "my_mnist_model.h5"
  shell:
    "python train.py --input {input} --output {output}"

rule eval:
  input:
    data="my_mnist.pkl",
    model="my_mnist_model.h5"
  output:
    "test.png"
  shell:
    "python eval.py --input {input.model} --data {input.data} --index={config[INDEX]}"
