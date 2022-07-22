# bulldozer_price_prediction_ml_project
my self project to predict bulldozer prices based on previous data 
I was not able to upload train.csv as file size was larger than 25mb and github only allows upto 25mb file. 
so here is the link of the kaggle dataset I used:-https://www.kaggle.com/c/bluebook-for-bulldozers/data
or you can directly use the train.csv file that I have uploaded in my gdrive :-https://drive.google.com/file/d/1k_7v2vM-DlY9LUk-7YgXy81pRIrPEGsW/view?usp=sharing

First I imported the important libraries like Numpy, Pandas, yfinance and pyplot from matplotlib
Then to import the data I changed the default directory to the directory where the data was located and imported it using read_csv.

# EDA
Now I checked the data for various things, like shape,min, max, missing values and the number of 
non numerical data and all.
Now, the saledate was not in the correct format so I used parse_dates(which will convert the string representation of date into a date time object) while reding the data and further use pandas api to extract important info.
I dropped the SalesID column because it wass all unique for every data so it would not have any contribution in training the model
Now I created p_data that act as a copy of the data so that I have the original data unchanged
Now as I have extracted the useful and individual components from the sale date column, I dont need it anymore, thus I dropped the saledate column from p_data

# Conversion of non numerical columns and data cleaning
Using pandas api and dtypes, I first printed all the column that have non numerical values 
and I converted them into categories using pandas categories
now I will see which numeric columns have missing values, and if there are any missing values I will fill them withe th median of the column.
Also we can fill them with mean, or mode but I will use median because median is more robust to changes and outliers.
here I also added a column to showcase that these numeric values were initially missing and I have added them through median

now we see that no numeric column has any missing value,

Thus, now we have to deal with the missing values in the categorical columns. So, as all of the data is now in the categorical form I would replace data with their categorical codes, so that all the data is finally in numerical form only.
I added +1 in the categorical code because by default python assigns the missing values with -1, so we added 1 so that the final categories start from 0 and not -1.
Finally now, the data is suitable to be trained on as it do not contain any missing values and any non numerical data

# Training the model
I imported the sklearn library and RandomForestRegressor model, and initially I only chose 10000 samples for training because first I will check the model on small sample size, if the model performs well, will continue with it. Because training and testing on the whole data set will consume a lot of time.
With the train_test_split I will split the data into a training set and a validation set, so that I can do hyperparameter tuning based on the results on the validation data set and improve our model.
as I plan to predict the sale price, I have seperated the data with label and feature. Thus y is a series with sale prices in it, while x contains the rest of the data
now I have split the data in such a way that 80% of the data is on the testing part and 20% on the validation part
Now I fit the model on the x_train and y_train and found the score on x_val and y_val to be 85%
now just with fitting the model on the training data set with 10000 samples, I have achieved a good score, now lets try to split in a different way and see if I can do better or not. But this is on the validation data, and not on the test data

Then I tried different ways of splitting the data into train and test, 
I split the data with saleyear as 1999 into validation set and the rest all as training set
and the with saleyear as 2009 into validation set and the rest all as training set

Then instead of splitting it into 2 parts and using the score, I used cross validation split, which actually is much better as it reduces the bias that can occur will split in a certain part, here lets split into 2 sections with 5 different samples
Thus, with the cross validation split I got a bit less score, but now this score is much more reliable and reflects the model in a better way

till Now, I have just looked at the score metric to judge the accuacy of the models, but there are much better ways to evaluate the model, such as r2-score, MEA(Mean absolute error), etc.

So, I created a function (show_metrics) that will give us the relative values of above mentioned metrics for that particular model.
The more is the r2_score the better is model, and the less the errors(MSE and MAE), the better is the model
Then I used RandomizedSearchCV that will help to choose better hyper parameters for the model from the provided grid
rf_grid is the dictionary of parameters and values that the model will check for and suggest the best parameters among them based on the number of iterations specified, To get this grid I searched on the internet about the parameter to tune and the values to check for

# testing on the actual data
I imported the test data from test.csv and did the same operation as I did on the train data to get them into same format before testing. 
Now again the test data has categorical columns and missing values, so I need to do the preprocessing in the same manner as for the training data, hence I created a preprocessing function that takes the data and process it to remove missing vales and convert all the data in the numerical form.
Now our test data do not have any missing values and any non numerical values, so I proceeded further.
Now,see that x_train has a column less than that of data_test, this is because, we added extra columns such as machinehours data is missing,etc to depict the missing values, so we need added these extra columns in the test data as well so that the comparison can be done.

# making predictions
Now we have a model based on 10000 samples, I will use it for the prediction now and compare with the original test data Also model is the RandomForestRegressor one and rs_model is the tuned model with RandomizedSearch CV
I named test_preds as the predicted value on test data by the randomForestRegressor model
and test_preds_rs as the predicted value on test data by the randomForestRegressor model tuned by RandomizedSearchCV
But the test data has an extra column("SalesID") so I dropped it to just include the target
column
Now using the model trained on 10000 samples we achieved an score of 86%, so then I used all of our data to train the model on
By doing 
