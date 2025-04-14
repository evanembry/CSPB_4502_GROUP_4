# CSPB_4502_GROUP_4


# Analytics in the Octagon – UFC Fight Prediction

## Team Members
- Evan Embry  


## Project Description

This project investigates whether historical UFC fight data can be effectively used to predict the outcome of bouts using machine learning techniques. I develop and compare several classification models, including Logistic Regression, Random Forest, and XGBoost—with comprehensive hyperparameter tuning via GridSearchCV—to understand which physical attributes and engineered comparative features best predict fight outcomes.

## Research Questions & Answers

- **What questions was I seeking to answer?**  
  - Can fighter attributes (height, reach, age, etc.) along with derived comparative metrics (reach, height, and age differences) accurately predict the winner of a UFC fight?
  - How do different modeling approaches (Logistic Regression, Random Forest, and XGBoost) compare in performance against traditional benchmarks such as bookmaker odds?

- **What did I find?**  
  - The tuned ensemble methods, especially Random Forest and XGBoost with hyperparameter tuning, captured the complex, nonlinear dynamics of fight data and achieved around 64.5% accuracy and a 70.5% F1 score.
  - Derived features (e.g., ReachDiff, HeightDiff, AgeDiff) are significant predictors, and our analysis provides actionable insights that can be applied for coaching, training, and betting strategy improvements.

## Application of This Knowledge

The insights from this project can be applied in multiple ways:
- **For Coaches and Fighters:** To optimize training regimens by focusing on the physical attributes that most significantly predict fight outcomes.
- **For Betting Markets:** To identify potential inefficiencies in bookmaker odds, aiding more informed wagering strategies.
- **For Broadcast and Analytics:** Providing data-driven insights during live events to enhance commentary and fan engagement.

## Additional Resources

- **Video Demonstration (must download .7z file) **  
  [Link to  Video Demonstration](Group_4_Analytics_In_The_Octagon_Part6_Video.7z)  
 

- **Final Project Paper:**  
  [Link Final Project Paper](#)  
  

## Source Code

All source code used for this project is available in this repository within main.py file. It includes:
- Data cleaning, preprocessing, and feature engineering scripts.
- Implementation and hyperparameter tuning of Logistic Regression, Random Forest, and XGBoost classifiers.
- Evaluation and result generation code.
- (Please run the terminal command `pip install -r requirements.txt` to install all required dependencies (e.g., xgboost) if they are not already installed on your machine.)

