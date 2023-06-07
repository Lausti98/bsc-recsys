# bsc-recsys

## How to execute experiments
### 1. Initialize virtual environment
To ensure the requirements for the project is aligned when executing the code, a virtual enviroment should be created, following the steps below: 
* ```$ python -m venv env```
* ```$ source env/bin/activate```

Once the virtual enviroment is activated, the requirements is installed by executing: 
* ```$ python -m pip install -r requirements.txt```
### 2. Run the experiments
Place the data folder from ERDA in place of ```./data``` folder.


Navigate into ``` src/benchmarks``` and execute files 

```python collaborative_filtering.py```

```python content_based.py```

```python hybrid.py```

Both files output the results for the algorithms for collaborative filtering and content based recommendation respectively.


