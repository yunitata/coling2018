The repository contains the source code to reproduce experiments from the COLING 2018 paper: Topic or Style? Exploring the Most Useful Features for Authorship Attribution.

Dependencies
------------
1. Python 2.7
2. Scikit Learn 0.18
3. Keras 1.1.1 (with Theano backend). By default, Keras will use TensorFlow as its tensor manipulation library. Please refers to the [Keras website] (https://keras.io/) [1] to configure the Keras backend.
4. Pandas
5. NLTK 3.0.4
6. Scipy 0.19.0
7. Seaborn 0.7.1
8. lda 1.0.4
9. Matplotlib 1.3.1

You can install all of these by running:

```pip install -r requirements.txt```


Cloning the repository
----------------------
```git clone https://github.com/yunitata/coling2018```


Preparing Data
--------------
1. All the dataset need to be requested directly from the author. Please refer the CCAT10 and CCAT50 to this [paper] (http://www.sciencedirect.com/science/article/pii/S0306457307001197) [2] while Judgment and IMDb62 to this [paper](http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00173) [3]. Please note that there are two version of IMDb62 datasets. In this experiment, we used the version which contains 62,000 movie reviews and 17,550 message board posts.
2. CCAT10, CCAT50 and IMDb62 datasets comes in the form of list of files per author. To make things easier, we merge all the documents from each of the author (for each of the dataset) into one csv file. It can be done with this following command:

  ```python data_prep.py folder_path csv_path "data_code"```

  CCAT10 and CCAT50 each comes with train and test folders, thus it will have separate train and test csv files.
  For example to prepare train and test data for CCAT10 data

  ```python data_prep.py "/home/C10train" "/home/C10_train.csv" "ccat"```

  ```python data_prep.py "/home/C10test" "/home/C10_test.csv" "ccat"```
  
  For IMDb62 dataset, it does not come with separate train/test set, to create the csv file: (fix this part)
  
  ```test ``` <br />
  
  Lastly, for Judgment dataset, it already comes in one .txt file, so no data preparation is needed.
