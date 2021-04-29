# FL_ClientSelection_FPAD

CSE 812 project

## Installation

```bash
numpy
pip install -r requirements.txt
```

## Usage

Note: The datasets used in this project will not be provided in this repo due to privacy restrictions.

First, we need to split our data into 80% train and 20% test and then create the appropriate data splits for each of our experiments. To do this, run the following command:

```
  python create_data_splits.py
```

Next, to establish a baseline with a non-federated learning environment, run the following python command which will train a Face PAD model on all of the aggregated data:

```
  python baseline.py
```

For many of the client selection algorithms for federeated learning, it is required to first train an initial model on the central server's dataset in order to compare with the local data distributions of each of the federated client nodes. To train this initial model, run:

```
  python fl_implementation_server.py
```

Finally, each experiment has it's own python script (e.g. fl_implementation_exp1.py). To run each experiment, set the appropriate flag for the client selection algorithm that you want to train the federated model with and simply run the script. For example, to run the first experiment:

```
  python fl_implementation_exp1.py
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
