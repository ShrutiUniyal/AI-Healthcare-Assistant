from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
# import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




app = Flask(__name__)

# Load the model and other necessary data
model_file = open(r'C:\Users\V\PycharmProjects\PythonProject\saved model\save.pkl', "rb")
rf_model = pickle.load(model_file)
model_file.close()

df = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\Training.csv')
sym_des = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\symptoms_df.csv')
precautions = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\precautions_df.csv')
workout = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\workout_df.csv')
description = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\description.csv')
medications = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\medications.csv')
diets = pd.read_csv(r'C:\Users\V\PycharmProjects\PythonProject\dataset\diets.csv')


# X = df.drop("prognosis", axis=1)
# y = df.prognosis
X = df.drop('prognosis', axis=1)
y = df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

# Disease data dictionary
dis_ld = medications["Disease"]
dis_lm = medications["Medication"]
dis_ldesc = description["Description"].to_list()
dis_lprec1 = precautions["Precaution_1"]
dis_lprec2 = precautions["Precaution_2"]
dis_lprec3 = precautions["Precaution_3"]
dis_lprec4 = precautions["Precaution_4"]
dis_lworkout = workout["workout"]
dis_ldiets = diets["Diet"]

disease_data = {}
for i in range(len(dis_ld)):
    disease_data[dis_ld[i]] = [dis_lm[i], dis_ldesc[i], dis_lprec1[i], dis_lprec2[i], dis_lprec3[i], dis_lprec4[i], dis_lworkout[i], dis_ldiets[i]]

# Reverse label encoding the labels
yy = label_encoder.inverse_transform(y)
dict = {}
for x in range(len(yy)):
    dict[yy[x]] = y[x]

# Function to recommend disease related info
def recomm(disease):

    for dd in disease_data:
        if dd == disease:
            medic_l = disease_data[dd][0]
            desc = disease_data[dd][1]
            prec1 = disease_data[dd][2]
            prec2 = disease_data[dd][3]
            prec3 = disease_data[dd][4]
            prec4 = disease_data[dd][5]
            work = disease_data[dd][6]
            diet = disease_data[dd][7]
    return medic_l, disease, desc, prec1, prec2, prec3, prec4, work, diet


def recomm_s(symptom_lst):
    symptom_feed = {b: (1 if b in symptom_lst else 0) for b in X.columns}
    feed_df = pd.DataFrame([symptom_feed])
    medd = rf_model.predict(feed_df)
    for key in dict:
        if dict[key] == medd[0]:
            return key


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symptoms_input = request.form.get("symptoms")
        input_symptoms = [symp.strip() for symp in symptoms_input.split(",")]

        disease = recomm_s(input_symptoms)
        if disease:
            medlist, disease, description, precaution1, precaution2, precaution3, precaution4, work, diet = recomm(
                disease)
            # Use session or URL parameters to pass results
            return redirect(url_for("index",
                                    disease=disease,
                                    description=description,
                                    medlist=medlist,
                                    precaution1=precaution1,
                                    precaution2=precaution2,
                                    precaution3=precaution3,
                                    precaution4=precaution4,
                                    work=work,
                                    diet=diet))
        else:
            flash("Could not predict disease.")  # Flash error message
            return redirect(url_for("index"))

    # GET request: Retrieve parameters from the URL
    disease = request.args.get("disease")
    description = request.args.get("description")
    medlist = request.args.get("medlist")
    precaution1 = request.args.get("precaution1")
    precaution2 = request.args.get("precaution2")
    precaution3 = request.args.get("precaution3")
    precaution4 = request.args.get("precaution4")
    work = request.args.get("work")
    diet = request.args.get("diet")

    return render_template("index.html",
                           disease=disease,
                           description=description,
                           medlist=medlist,
                           precaution1=precaution1,
                           precaution2=precaution2,
                           precaution3=precaution3,
                           precaution4=precaution4,
                           work=work,
                           diet=diet)


if __name__ == "__main__":
    app.run(debug=True)
