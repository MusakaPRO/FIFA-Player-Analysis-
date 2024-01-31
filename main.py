import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = 'E:/SAP/fifa_data3.xlsx'
fifa_data = pd.read_excel(file_path)


# Function to convert height from feet'inches" to inches
def height_to_inches(height):
    if isinstance(height, str):
        feet, inches = height.split("'")
        return int(feet) * 12 + int(inches.replace('"', ''))
    else:
        return np.nan  # Return NaN if the value is not a string


# Convert 'Height' and 'Weight'
fifa_data['Height'] = fifa_data['Height'].apply(height_to_inches)
fifa_data['Weight'] = fifa_data['Weight'].str.replace('lbs', '', regex=True).astype('float')

# Dropping irrelevant columns, rows with missing values, and handling NaN values
fifa_data_cleaned = fifa_data.drop(
    columns=['Unnamed: 0', 'ID', 'Name', 'Nationality', 'Club', 'Preferred Foot', 'Position']).dropna()

# Basic statistical summary of the dataset
summary_statistics = fifa_data_cleaned.describe()

# Calculating the correlation matrix
correlation_matrix = fifa_data_cleaned.corr()

# Splitting the heatmap into smaller sections for detailed visualization
# Heatmap 1: Correlation of player attributes like 'Age', 'Overall', 'Potential', 'International Reputation'
attributes_1 = ['Value', 'Age', 'Overall', 'Potential', 'International Reputation']
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix.loc[attributes_1, attributes_1], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation - Player Attributes')
plt.savefig('correlation_heatmap_attributes.png')

# Heatmap 2: Correlation of player skills like 'Finishing', 'Dribbling', 'BallControl', 'Acceleration'
skills_1 = ['Value', 'Finishing', 'Dribbling', 'BallControl', 'Acceleration']
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix.loc[skills_1, skills_1], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation - Player Skills')
plt.savefig('correlation_heatmap_skills.png')

# Heatmap 3: Correlation of player physical traits like 'Strength', 'Stamina', 'Jumping', 'Agility'
physical_1 = ['Value', 'Strength', 'Stamina', 'Jumping', 'Agility']
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix.loc[physical_1, physical_1], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation - Player Physical Traits')
plt.savefig('correlation_heatmap_physical.png')

# Selecting features and target for the model
features = fifa_data_cleaned.drop(columns=['Value'])
target = fifa_data_cleaned['Value']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print('Mean Squared Error:', mse)
print('R² Score:', r2)

# Plotting Predicted vs Actual Values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, predictions, alpha=0.3)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values for Player Market Value')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.savefig('predicted_vs_actual_values.png')

# Extracting the absolute values of correlations with 'Value' and sorting them
value_correlation_abs = correlation_matrix['Value'].abs().sort_values(ascending=False)

# Selecting the top factors influencing player's value
top_factors = value_correlation_abs[1:11]  # Adjust the number of factors as needed

# Plotting the top factors influencing the player's value
plt.figure(figsize=(10, 6))
top_factors.plot(kind='bar')
plt.title('Top Factors Influencing Player\'s Market Value')
plt.xlabel('Factors')
plt.ylabel('Absolute Correlation with Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_factors_influencing_value.png')

# Identifying the top 10 highest paid players and their clubs and values
top_paid_players = fifa_data.sort_values(by='Wage', ascending=False).head(10)
top_paid_players_info = top_paid_players[['Name', 'Club', 'Wage', 'Value']]

# Creating a diagram (bar chart) for the top 10 highest paid players
plt.figure(figsize=(12, 8))
sns.barplot(x='Wage', y='Name', data=top_paid_players_info, hue='Club')
plt.title('Top 10 Highest Paid Players and Their Clubs')
plt.xlabel('Wage (in thousands $)')
plt.ylabel('Player Name')
plt.legend(title='Club', loc='lower right')
plt.tight_layout()
plt.savefig('top_10_highest_paid_players.png')

# Correlation between 'Wage' and 'Value'
correlation_wage_value = fifa_data[['Wage', 'Value']].corr()

# Grouping by 'Nationality' and calculating mean 'Wage', 'Value', and 'Potential'
nationality_stats = fifa_data.groupby('Nationality')[['Wage', 'Value', 'Potential']].mean()

# Sorting nationalities by average 'Wage' and 'Potential'
highest_paid_nationalities = nationality_stats['Wage'].sort_values(ascending=False).head()
lowest_paid_nationalities = nationality_stats['Wage'].sort_values(ascending=True).head()
highest_potential_nationalities = nationality_stats['Potential'].sort_values(ascending=False).head()

# Plotting results for 'Wage', 'Value', and 'Potential' by 'Nationality'
# Nationalities with Highest Average Wage
plt.figure(figsize=(10, 6))
highest_paid_nationalities.plot(kind='bar', color='green')
plt.title('Top 5 Nationalities with Highest Average Wage')
plt.ylabel('Average Wage')
plt.xlabel('Nationality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('highest_paid_nationalities.png')

# Nationalities with Lowest Average Wage
plt.figure(figsize=(10, 6))
lowest_paid_nationalities.plot(kind='bar', color='red')
plt.title('Top 5 Nationalities with Lowest Average Wage')
plt.ylabel('Average Wage')
plt.xlabel('Nationality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('lowest_paid_nationalities.png')

# Nationalities with Highest Average Potential
plt.figure(figsize=(10, 6))
highest_potential_nationalities.plot(kind='bar', color='blue')
plt.title('Top 5 Nationalities with Highest Average Potential')
plt.ylabel('Average Potential')
plt.xlabel('Nationality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('highest_potential_nationalities.png')

# Displaying the results
print("\nNationalities with Highest Average Wage:\n", highest_paid_nationalities)
print("\nNationalities with Lowest Average Wage:\n", lowest_paid_nationalities)
print("\nNationalities with Highest Average Potential:\n", highest_potential_nationalities)
