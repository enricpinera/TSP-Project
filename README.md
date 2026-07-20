# Travelling Salesman Problem: Heuristics and Neural Models

This project compares classical heuristics and neural approaches for the
Travelling Salesman Problem (TSP). It includes implementations for constructive
heuristics, local search, metaheuristics, a Graph Convolutional Network (GCN)
model and a Transformer model, together with generated result files and summary
validation utilities.

## Project Structure

- `Heuristics.py`: evaluates nearest neighbor, greedy, Christofides, 2-opt,
  3-opt, simulated annealing and threshold accepting.
- `GCN.py`: trains the GCN-based TSP model.
- `Transformer.py`: trains the Transformer-based TSP model.
- `GCNTester.py`: evaluates saved GCN checkpoints.
- `TransformerTester.py`: evaluates saved Transformer checkpoints.
- `ResultsValidator.py`: validates result CSV files and generates summary CSVs.
- `requirements.txt`: lists the Python dependencies needed to run the project.
- `GCNReports.txt` and `TransformersReports.txt`: training reports for the
  neural models.
- `Models/`: saved model checkpoints.
- `Results/`: experiment outputs and summary files.
- `Visualizations/`: scripts used to generate plots and visual outputs.
- `tsp-data/`: expected dataset folder. It is required to run the scripts, but
  it is not included in the repository because of its size.

## Requirements

The project uses Python 3.10+ and the dependencies listed in
`requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

It is recommended to use a GPU.

## Dataset

The dataset is required to run the training and evaluation scripts, but it is
not included in this repository because it is too large for GitHub.

> [!IMPORTANT]
> Before running the project, download the dataset and place it exactly as shown
> below.

Download link:
[TSP dataset](https://drive.google.com/file/d/1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp/view)

Extract the `.tar.gz` file, create a folder named `tsp-data` in the root of this project
and place the extracted `.txt` files inside of it.

The final structure must be:

```text
tsp-data/tsp10_train_concorde.txt
tsp-data/tsp10_val_concorde.txt
tsp-data/tsp10_test_concorde.txt
tsp-data/tsp20_train_concorde.txt
tsp-data/tsp20_val_concorde.txt
tsp-data/tsp20_test_concorde.txt
tsp-data/tsp30_train_concorde.txt
tsp-data/tsp30_val_concorde.txt
tsp-data/tsp30_test_concorde.txt
```

## Experiments

Run heuristic evaluation:

```bash
python Heuristics.py
```

Train the GCN model:

```bash
python GCN.py
```

Train the Transformer model:

```bash
python Transformer.py
```

Evaluate saved checkpoints:

```bash
python GCNTester.py
python TransformerTester.py
```

Validate and summarize results:

```bash
python ResultsValidator.py
```

## Outputs

Experiment CSV files are stored in `Results/`. The validator produces summary
files such as:

- `Results/summary_tsp10.csv`
- `Results/summary_tsp20.csv`
- `Results/summary_tsp30.csv`

Saved model checkpoints are stored in `Models/`.

## Notes

- The scripts expose the main experiment settings near the bottom of each file.
- Set `num_nodes` to `10`, `20` or `30` depending on the experiment to run.
- Result files currently included in this repository correspond to the final
  experiments prepared for submission.
