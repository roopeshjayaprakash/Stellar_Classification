from enum import auto
import streamlit as st
from sklearn.svm import SVC
from sklearn.metrics import pair_confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
import pandas as pd 
import numpy as np 
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder #Data preprocessing
from sklearn.preprocessing import OneHotEncoder, label_binarize, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report #Model evaluation
from imblearn.over_sampling import SMOTE # for handling imbalanced classes 
from sklearn.model_selection import StratifiedKFold, GridSearchCV 
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #implementing RFC's 
from sklearn.svm import SVC #implementing SVC's
from sklearn.tree import DecisionTreeClassifier #implementing DTC'c
from sklearn.impute import SimpleImputer # for null value imputation
import xgboost as xgb 
from xgboost import XGBClassifier #implementing XGBC
from sklearn.naive_bayes import MultinomialNB #implementing MNBC
from sklearn.neighbors import KNeighborsClassifier #implementing KNC
from sklearn.naive_bayes import GaussianNB #implementing NBC
from sklearn.compose import ColumnTransformer #combining the pipeline
from sklearn import set_config # Displaying the pipeline
from astropy.time import Time #for date manipulation
from statsmodels.tsa.seasonal import seasonal_decompose #for time series analysis
import matplotlib.dates as mdates#for time series analysis
import warnings
warnings.filterwarnings("ignore")
from scipy.stats.mstats import winsorize

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call function to apply styles
load_css("style.css")

def main():
    st.markdown(
    """
    <div style="
        background-color: rgba(0, 0, 0, 0.9); 
        padding: 20px; 
        border-radius: 10px;
        text-align: left;
        font-size: 40px;
        color: white;
        font-weight: bold;
        width: 100%;
    "> Stellar Classification Web App </div> """, unsafe_allow_html=True)
    st.sidebar.title("Stellar Classification Web App")
    st.sidebar.markdown("What are you looking at? A Galaxy?🌌 A Quasar?✨ A Star?⭐")

    #Function for loading the dataset and since All the columns in the data set are categorical we perform label encoding'''
    @st.cache_data(persist=True) #Using this decorator to cache the output which we run 
    def load_data():
        st.markdown(
        '<div style="background-color: white; padding: 10px; border-radius: 5px;">'
        '<p>Project Objective: The classification of celestial objects—such as stars, galaxies, and quasars—is a fundamental task in Astrophysics and Observational astronomy. '
        'Given the vast amount of Astronomical data collected through large-scale sky surveys, automated classification using machine learning has become increasingly important. '
        'This project leverages machine learning techniques to classify celestial objects based on key astronomical features extracted from telescopic observations.</p>'
        '</div>', unsafe_allow_html=True)
        df=pd.read_csv('star_classification.csv') 
        df_ids=[i for i in df.columns if i.__contains__('ID')]#remove cam_col too later
        for i in df_ids:
            df=df.drop([i], axis=1)
            df=df.drop(['cam_col', 'r', 'plate', 'MJD', 'g', 'z'], axis=1)#all unnecessary columns
            return df
    
    
    @st.cache_data(persist=True)
    def split(df):
        y = df['class']

        # Extract features
        X = df.drop(columns=['class'])

        for i in X.columns:
            # Detect outliers using IQR method
            Q1 = X[i].quantile(0.25)
            Q3 = X[i].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = (X[i] < lower_bound) | (X[i] > upper_bound)

            # Calculate mean without outliers, handling NaN cases
            mean_value = X.loc[~outliers, i].mean()
            mean_value = mean_value if pd.notna(mean_value) else X[i].mean()  # Fallback

            # Replace outliers with mean
            X.loc[outliers, i] = mean_value

        columns_to_winsorize = ['u', 'i', 'redshift']

        X_winsorized = X.copy()

        # Apply Winsorization
        for col in columns_to_winsorize:
            X_winsorized[col] = winsorize(X[col], limits=[0.05, 0.05])  

        transformer=PowerTransformer('yeo-johnson')#We apply yeo-johnson because it contains negetive values
        X_transformed=pd.DataFrame(transformer.fit_transform(X_winsorized), columns=X.columns)

        X=X_transformed.copy()

        label = LabelEncoder()
        y = pd.Series(label.fit_transform(y), name="class")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        
        X_train=X_train_scaled
        X_test=X_test_scaled
        
        return X_train, X_test, y_train, y_test
    
    def plot_metrics(metrics_list, model, X_test, y_test, class_names, y_pred):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            y_test_bin = label_binarize(y_test, classes=range(len(class_names)))  # Convert to binary labels
            y_score = model.predict_proba(X_test)  # Get probability scores

            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")

            ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Multiclass ROC Curve")
            ax.legend()
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            y_test_bin = label_binarize(y_test, classes=range(len(class_names)))  
            y_score = model.predict_proba(X_test)  

            for i, class_name in enumerate(class_names):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
                pr_display.plot(ax=ax, label=f"Class {class_name}")

            ax.set_title("Multiclass Precision-Recall Curve")
            st.pyplot(fig)

    #Checkbox widget showing raw dataframe on mainpage
    @st.cache_data(persist=True) #Using this decorator to cache the output which we run 
    def raw_data():
        df=pd.read_csv("/home/roopesh-j/notebooks/BIA-20241227T162415Z-001/BIA/Stellar Object Classification Project/star_classification.csv")
        return df
    
    st.sidebar.header("Menu")
    show_raw_data = st.sidebar.checkbox("Show Raw Data", False)

    descriptions = """
    <div style="background-color: white; padding: 10px; border-radius: 5px;">
        <p><strong>1. obj_ID</strong>: Object Identifier, the unique value that identifies the object in the image catalogue used by the CAS</p>
        <p><strong>2. alpha</strong>: Right Ascension angle (at J2000 epochs)</p>
        <p><strong>3. delta</strong>: Declination angle (at J2000 epoch)</p>
        <p><strong>4. u</strong>: Ultraviolet filter in the photometric system</p>
        <p><strong>5. g</strong>: Green filter in the photometric system</p>
        <p><strong>6. r</strong>: Red filter in the photometric system</p>
        <p><strong>7. i</strong>: Near Infrared filter in the photometric system</p>
        <p><strong>8. z</strong>: Infrared filter in the photometric system</p>
        <p><strong>9. run_ID</strong>: Run Number used to identify the specific scan</p>
        <p><strong>10. rereun_ID</strong>: Rerun Number to specify how the image was processed</p>
        <p><strong>11. cam_col</strong>: Camera column to identify the scanline within the run</p>
        <p><strong>12. field_ID</strong>: Field number to identify each field</p>
        <p><strong>13. spec_obj_ID</strong>: Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)</p>
        <p><strong>14. class</strong>: Object class (galaxy, star or quasar object)</p>
        <p><strong>15. redshift</strong>: Redshift value based on the increase in wavelength</p>
        <p><strong>16. plate_ID</strong>: Plate ID, identifies each plate in SDSS</p>
        <p><strong>17. MJD</strong>: Modified Julian Date, used to indicate when a given piece of SDSS data was taken</p>
        <p><strong>18. fiber_ID</strong>: Fiber ID that identifies the fiber that pointed the light at the focal plane in each observation</p>
    </div>
    """

    if show_raw_data:
        st.subheader("Dataset")
        st.write(raw_data())  # Ensure `raw_data()` is defined
        st.subheader("Feature Description")
        st.markdown(descriptions, unsafe_allow_html=True)
        st.subheader("Right Ascension and Declination")
        st.image("Right-ascension-and-declination.png", caption="Image Credits: skyandtelescope.org", use_container_width=True)
        st.subheader("Sloan Digital Sky Survey (SDSS) Photometric System - Filters")
        st.image("white-spectrum-family-sloan.jpeg", caption="Image Credits: altairastro.com", use_container_width=True)
    
    @st.cache_data(persist=True)
    def EDA():
        df = pd.read_csv("/home/roopesh-j/notebooks/BIA-20241227T162415Z-001/BIA/Stellar Object Classification Project/star_classification.csv")
        profile = ProfileReport(df, title="Profile Report")
        return profile.to_html() 

    if st.sidebar.checkbox("Profile Report", False):
        st.subheader("Profile Report")
        st.components.v1.html(EDA(), height=800, scrolling=True)  # Embed the report properly
    
    df=load_data()
    x_train, x_test, y_train, y_test=split(df)
    class_names=['GALAXY', 'QSO', 'STAR']
    st.sidebar.subheader("Choose Classifier") #User can select the type of classifier they want to use
    Classifier=st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)",
                                                    "Random Forest", 
                                                    "Decision Tree",
                                                    "Naive Bayes",
                                                    "K Nearest Neighbours"))

    if Classifier=='Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularisation Parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel=st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))


        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Loading Support Vector Machine (SVM) Results.......")
            model=SVC(probability=True)
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)
            accuracy=model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, average='macro')  # Micro-averaged
            recall = recall_score(y_test, y_pred, average='micro')  # Micro-averaged
            f1score=f1_score(y_test, y_pred, average='macro') 
            st.markdown(f"""
            <div style="border:2px solid #4CAF50; padding:10px; border-radius:10px; background-color:#f4f4f4; text-align:center;">
                <h4 style="color:#00000;">Model Performance</h4>
                <p><b>Accuracy:</b> {accuracy:.4f}</p>
                <p><b>Precision:</b> {precision:.4f}</p>
                <p><b>Recall:</b> {recall:.4f}</p>
                <p><b>F1 Score:</b> {f1score:.4f}</p>
            </div>""",unsafe_allow_html=True)


            plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)#User selected metrics from above
    
    if Classifier=='Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimator = st.sidebar.slider("The number of trees in the forest", min_value=100, 
                        max_value=5000, step=10, key="n_estimator")
        max_depth=st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = bool(st.sidebar.radio("Bootstrap samples when building trees",('True','False'), key='bootstrap'))
        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Loading Random Forest Classifier Results.......")
            model=RandomForestClassifier()
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)
            accuracy=model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, average='macro')  # Micro-averaged
            recall = recall_score(y_test, y_pred, average='micro')  # Micro-averaged
            f1score=f1_score(y_test, y_pred, average='macro') 
            st.markdown(f"""
            <div style="border:2px solid #4CAF50; padding:10px; border-radius:10px; background-color:#f4f4f4; text-align:center;">
                <h4 style="color:#4CAF50;">Model Performance</h4>
                <p><b>Accuracy:</b> {accuracy:.4f}</p>
                <p><b>Precision:</b> {precision:.4f}</p>
                <p><b>Recall:</b> {recall:.4f}</p>
                <p><b>F1 Score:</b> {f1score:.4f}</p>
            </div>""",unsafe_allow_html=True)
            plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)#User selected metrics from above
    
    if Classifier=='Decision Tree':
        st.sidebar.subheader("Model Hyperparameters")
        max_depth=st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        classifier__criterion = bool(st.sidebar.radio("Classification Criterion",('Gini','Entropy'), key='classifier__criterion'))
        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Loading Decision Tree Results.......")
            model=DecisionTreeClassifier()
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)
            accuracy=model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, average='macro')  # Micro-averaged
            recall = recall_score(y_test, y_pred, average='micro')  # Micro-averaged
            f1score=f1_score(y_test, y_pred, average='macro')
            st.markdown(f"""
            <div style="border:2px solid #4CAF50; padding:10px; border-radius:10px; background-color:#f4f4f4; text-align:center;">
                <h4 style="color:#4CAF50;">Model Performance</h4>
                <p><b>Accuracy:</b> {accuracy:.4f}</p>
                <p><b>Precision:</b> {precision:.4f}</p>
                <p><b>Recall:</b> {recall:.4f}</p>
                <p><b>F1 Score:</b> {f1score:.4f}</p>
            </div>""",unsafe_allow_html=True)
            plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)#User selected metrics from above

    if Classifier=='Naive Bayes':
        st.sidebar.subheader("Model has no Hyperparameters")
        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Loading Naive Bayes Results.......")
            model=GaussianNB()
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)
            accuracy=model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, average='macro')  # Micro-averaged
            recall = recall_score(y_test, y_pred, average='micro')  # Micro-averaged
            f1score=f1_score(y_test, y_pred, average='macro') 
            st.markdown(f"""
            <div style="border:2px solid #4CAF50; padding:10px; border-radius:10px; background-color:#f4f4f4; text-align:center;">
                <h4 style="color:#4CAF50;">Model Performance</h4>
                <p><b>Accuracy:</b> {accuracy:.4f}</p>
                <p><b>Precision:</b> {precision:.4f}</p>
                <p><b>Recall:</b> {recall:.4f}</p>
                <p><b>F1 Score:</b> {f1score:.4f}</p>
            </div>""",unsafe_allow_html=True)
            plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)#User selected metrics from above

    if Classifier=='K Nearest Neighbours':
        st.sidebar.subheader("Model Hyperparameters")
        classifier__weights = st.sidebar.radio("Number of Neighbours",('uniform', 'distance'), key='classifier__weights')
        classifier__n_neighbors = st.sidebar.radio("Number of Neighbours",(3,5,7,10), key='classifier__n_neighbors')
        metrics=st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Loading K Nearest Neighbours (KNN) Results.......")
            model=KNeighborsClassifier()
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)
            accuracy=model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, average='macro')  # Micro-averaged
            recall = recall_score(y_test, y_pred, average='micro')  # Micro-averaged
            f1score=f1_score(y_test, y_pred, average='macro')
            st.markdown(f"""
            <div style="border:2px solidrgb(0, 3, 0); padding:10px; border-radius:10px; background-color:#f4f4f4; text-align:center;">
                <h4 style="color:black;">Model Performance</h4>
                <p style="color:black;"><b>Accuracy:</b> {accuracy:.4f}</p>
                <p style="color:black;"><b>Precision:</b> {precision:.4f}</p>
                <p style="color:black;"><b>Recall:</b> {recall:.4f}</p>
                <p style="color:black;"><b>F1 Score:</b> {f1score:.4f}</p>
            </div>""",unsafe_allow_html=True)
            plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)#User selected metrics from above

#Future scope - K Fold Cross Validation, using .predict() function, 
if __name__ == '__main__':
    main()