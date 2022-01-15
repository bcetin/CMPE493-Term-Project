Welcome to our CMPE493 Term Project
by Maral Dicle Maral, Hazer Babür, Burak Çetin

To use the code you need the requirements installed in any python 3.8 environment:

pip install -r requirements.txt

There are 3 main code pieces. Each of them can be run by "python filename.py"

preprocess.py creates a .pkl file that contains the document frequencies to be used by the training code.

training.py creates a .pkl file that contains the model to be used by the test code.

test.py creates the prediction.csv file that contains the document frequencies to be graded by the code provided by the Biocreative Track.

As we tried different parameters, in the training.py file there is a section that can be used to change those parameters labeled "PARAMETERS FOR DOING ANALYSIS".
The variables below that section can be changed to change the prediction or training method.