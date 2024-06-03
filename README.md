# Z-MachineLearningLibrary
Our Personal Machine Learning Library

This is a Machine Learning library that Abderrahmane Baidoune, Imane Rahali I would like to build in order to further our understanding of the algorithms and implement them from scratch with the help of numpy, no more.

Thus, optimization is not a concern to us, nor is documentation or code readability. Having said that, we will, and have tried to devote some effort to it !


## Features

- **Machine Learning Algorithms**: Implementations of various machine learning algorithms.
- **No Dependencies**: Relies solely on numpy, avoiding other external dependencies.
- **Educational Focus**: Emphasis on understanding the underlying mechanics of algorithms.

## Requirements

- Python 3.6 or higher
- numpy

## Installation

You can install Z-MachineLearningLibrary via pip:

```bash
pip install ZAIScikit
```

Or clone the repository and install it locally:

```bash
git clone https://github.com/yourusername/ZAIScikit.git
cd ZAIScikit
pip install -e .
```

## Future Improvements and additions

<h5 style="color:green"> All Models' features need to be 2D Arrays even if there is one feature </h3>
<ul>
<li> Better OOP Design and Redundancy Omitting
<li> To be implemented :
    <ul>
        <li> DBSCAN and HDBSCAN
        <li> UMAP
        <li> Reinforcement Learning
        <li> AlphaZero
        <li> Factorization Methods
        <li> Convolutional Neural Networks
        <li> RNN + LSTM
        <li> Transformers
    </ul>
<li> Needs Better Implementations :
    <ul>
        <li> Faster BallTree / KDTree Algorithms for KNN
    </ul>
</ul>


## Performance Comparison
Below are some comparisons of our implementations with scikit-learn's implementations.

##### Decision Tree Regressor Performance

![Decision Tree Comparison](decisiontreeperf.png)

### Gaussian NB Performance

![Gaussian NB Comparison](gaussiannbperf.png)

### KMeans Clustering Performance

![KMeans Clustering Comparison](kmeansperf.png)