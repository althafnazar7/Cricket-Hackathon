# Cricket-Hackathon
Predicts the daily score of an IPL match at the end of 6 overs.<br/>
The **'InputFile'** contains the different features of the IPL match on a particular day like the venue, batting team, bowling team, players etc.
The contents of **InputFile** can be changed accordingly.<br/>
**'Trainer.py'** trains the model and set wieghts to the features from a dataset containing over 10 years of IPL data.<br/>
**'Predictor.py'** takes input from 'Input_File' and returns an output score based on the trained model<br/>
**'main.py'** calls **'Predictor.py'** and displays the runs predicted.<br/>
**RUN 'main.py' after modifying 'InputFile' as required**<br/>
