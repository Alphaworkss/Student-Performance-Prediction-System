# File: StudentGradePrediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------
# 1. LOAD OR CREATE DATASET (preserve existing dataset)
# -----------------------------------------------------
dataset_file = "student_dataset.csv"

if os.path.exists(dataset_file):
    df = pd.read_csv(dataset_file)
    print(f"Loaded existing dataset with {len(df)} students.")
else:
    # Create initial dataset
    data = {
        "Attendance": np.random.randint(50, 101, 200),
        "StudyHours": np.random.randint(1, 8, 200),
        "PreviousExamScore": np.random.randint(30, 101, 200),
        "AssignmentsCompleted": np.random.randint(50, 101, 200),
        "ExtraActivities": np.random.randint(0, 2, 200)  # 0 = No, 1 = Yes
    }

    df = pd.DataFrame(data)
    df["RollNumber"] = range(1, len(df) + 1)

    # Rearrange columns
    df = df[[
        "RollNumber", "Attendance", "StudyHours", "PreviousExamScore",
        "AssignmentsCompleted", "ExtraActivities"
    ]]

    # Assign grades
    def assign_grade(row):
        score = (row["Attendance"] * 0.15 +
                 row["StudyHours"] * 0.10 +
                 row["PreviousExamScore"] * 0.50 +
                 row["AssignmentsCompleted"] * 0.20 +
                 row["ExtraActivities"] * 0.05)

        if score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    df["FinalGrade"] = df.apply(assign_grade, axis=1)
    df.to_csv(dataset_file, index=False)
    print("Dataset created and saved.")

# -----------------------------------------------------
# 2. MODEL TRAINING
# -----------------------------------------------------
features = ["Attendance", "StudyHours", "PreviousExamScore",
            "AssignmentsCompleted", "ExtraActivities"]
X = df[features]
y = df["FinalGrade"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Print test accuracy
test_acc = model.score(X_test, y_test)
print(f"Model Test Accuracy: {test_acc*100:.2f}%")

# -----------------------------------------------------
# 3. GRAPHICAL DISPLAY FUNCTION
# -----------------------------------------------------
def show_graph(student_features, grade):
    feature_names = features
    colors = {
        "A": "green",
        "B": "blue",
        "C": "orange",
        "D": "purple",
        "F": "red"
    }
    plt.figure(figsize=(8,5))
    plt.bar(feature_names, student_features, color='skyblue')
    plt.title(f"Student Features and Assigned Grade: {grade}", fontsize=14)
    plt.ylabel("Values")
    plt.xlabel("Features")

    # Add grade color indicator
    plt.gca().set_facecolor(colors.get(grade, 'gray'))
    plt.show()

# -----------------------------------------------------
# 4. INPUT VALIDATION FUNCTION
# -----------------------------------------------------
def get_valid_input(prompt, min_val, max_val, type_=int):
    while True:
        try:
            val = type_(input(prompt))
            if val < min_val or val > max_val:
                print(f" Value must be between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print(f" Invalid input. Enter a {type_.__name__}.")

# -----------------------------------------------------
# 5. PREDICTION FUNCTIONS
# -----------------------------------------------------
def predict_from_input():
    print("\nEnter student details:")
    att = get_valid_input("Attendance (50–100): ", 50, 100)
    sh = get_valid_input("Study Hours (1–7): ", 1, 7)
    prev = get_valid_input("Previous Exam Score (30–100): ", 30, 100)
    ac = get_valid_input("Assignments Completed (50–100): ", 50, 100)
    ea = get_valid_input("Extra Activities (0=No, 1=Yes): ", 0, 1)

    features_df = pd.DataFrame([[att, sh, prev, ac, ea]], columns=features)
    pred = model.predict(features_df)
    grade = le.inverse_transform(pred)[0]

    print(f"\nPredicted Grade: {grade}")
    show_graph([att, sh, prev, ac, ea], grade)

def predict_from_roll():
    roll = get_valid_input("\nEnter Roll Number: ", 1, int(df["RollNumber"].max()))
    if roll not in df["RollNumber"].values:
        print("Roll number not found!")
        return

    student = df[df["RollNumber"] == roll].iloc[0]
    features_df = pd.DataFrame([[
        student["Attendance"],
        student["StudyHours"],
        student["PreviousExamScore"],
        student["AssignmentsCompleted"],
        student["ExtraActivities"]
    ]], columns=features)
    
    pred = model.predict(features_df)
    grade = le.inverse_transform(pred)[0]

    print("\nStudent Found:")
    print(student)
    print(f"\nPredicted Grade: {grade}")
    show_graph([
        student["Attendance"],
        student["StudyHours"],
        student["PreviousExamScore"],
        student["AssignmentsCompleted"],
        student["ExtraActivities"]
    ], grade)

# -----------------------------------------------------
# 6. MAIN MENU LOOP
# -----------------------------------------------------
while True:
    print("\n===============================")
    print(" STUDENT GRADE PREDICTION MENU")
    print("===============================")
    print("1. Predict grade by entering student features")
    print("2. Predict grade using Roll Number")
    print("3. Exit")
    
    choice = input("\nEnter your choice: ")

    if choice == "1":
        predict_from_input()
    elif choice == "2":
        predict_from_roll()
    elif choice == "3":
        print("\nExiting program... Goodbye!")
        break
    else:
        print(" Invalid choice, please try again!")
