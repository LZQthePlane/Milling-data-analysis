# Milling-data-analysis
#### Using SVR to predict the milling-blade wearing value based on basic parameters, and output as .xml format.

 - `data.csv` : Contains the dataset for training.
 - `svr_train.xml` :Script of training, using *Support Vector Regression* in `SKlearn`. Finally get a score of **90%**.
 - `svr_predict.xml` : Script of predicting, and output the result in .xml format. **Note**: the predict input data are generated randonmly.
 - `SVR_model.pkl` : The pre-trained model got after training process, and in predicting process be loaded.
