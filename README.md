# League of Legends Data Analysis/Match Prediction Project 
![League Of Legends Picture](https://github.com/NikhilInampudi/LeagueOfLegends-DataScience/blob/e841a20b80d31004d3a1e23303e551cd497f2407/League%20Of%20Legends%20Picture.jpg)

## Table of Contents
- [Overview](#overview)
- [Tools Used](#tools-used)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Data Extraction](#data-extraction)
- [Data Transformation](#data-transformation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Machine Learning](#machine-learning)
- [Results and Insights](#results-and-insights)

## Overview
As an avid *League of Legends* player and data science enthusiast, I wanted to combine my passion for gaming with my interest in data analysis. This project aims to provide actionable insights into player performance, champion effectiveness, and match outcomes, helping players like me optimize their strategies and improve their gameplay. This project focuses on analyzing player data by using the Riot Games API. The goal is to extract, process, and visualize various player statistics to gain insights into player performance, champion usage, and other key metrics. The project also utilizes certain fields such as kills and assists to predict whether a match outcome will be win or loss. The project is implemented in a Jupyter Notebook and leverages Python libraries such as requests, pandas, matplotlib, seaborn, and sci-kit learn for data extraction, manipulation, visualization, and machine learning.

## Tools Used
- Visual Studio Code
- Jupyter Notebook
- Python Libaries:
  - Requests
  - Pandas
  - Matplotlib
  - Seaborn
  - Sci-kit Learn

## Problem Statement
How can I analyze League of Legends match data to optimize my in-game strategies and what habits do I tend to have on specific champions?

## Project Structure
The project is divided into several sections, each focusing on a different aspect of data science:

1. **Data Extraction:** Using the Riot Games API to fetch player data, including match history, champion performance, and other relevant statistics.

2. **Data Transformation:** Converting the extracted data into a dataframe object and changing data types so format is suitable for analysis and predictive modeling.

3. **Exploratory Data Analysis (EDA):** Performing initial analysis to understand the data distribution, identify trends, and generate insights. Creating visual representations of the data to better understand player performance, champion usage, and win/loss ratio. 

4. **Feature Engineering:** Creating new features and scaling relevant features to optimize prediction outcomes and ensure data consistency. 
   
5. **Machine Learning:** Applying logistic regression algorithm to create meaningful conclusions from the data and seeing if we can predict match outcome.

## Data Extraction
The project begins by using the Riot Games API to fetch player data. The API provides access to various endpoints that allow us to retrieve information about players such as PUUID (Player Universally Unique Identifier), match IDS, and match data. The key steps in this section include:

- **API Key Authentication:** Securely accessing the API using an API key.
- **Fetching Player Data:** Retrieving player-specific data from 20 most recent matches such as champion performance, win rates, and other relevant statistics.
- **Storing Data:** Saving the fetched data into variables for further processing.

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Adding dependency to make a request to the API 
import requests

#Prompting user for API Key and storing in variable 
api_key = input('What is your API key? ')

url = 'https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/Phant%C3%B3m/NA1'

# Set up headers with the API key
headers = {
    'X-Riot-Token': api_key
}
    
# Send the GET request to the API
response = requests.get(url, headers=headers)

# Check if the response status is 200 (success)
if response.status_code == 200:
    # Parse the JSON response into a Python dictionary
    data = response.json()
    print(data)  # Return or print the JSON object
else:
    # If there's an error, print the status code and error message
    print(f"Error: {response.status_code} - {response.text}")

data
```
This data variable stores the PUUID (Player Universally Unique Identifier), which is then used to retrieve 20 match IDs, followed by their corresponding match data.

### Functions
When building our project, we may need to collect extensive player data. To streamline this process and improve code reusability, we can create functions that make the code more manageable. I begin by converting some API calls into functions, enhancing both efficiency and readability. These functions include:

(API Key is a default parameter to use every function as it is necessary for making public calls to the Riot API)

**Function #1 allows us to get the PUUID by passing Summoner Name, Tagline, and Region**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#This function returns puuid by using summoner name, player tagline, and region as parameters
def get_puuid(summoner_name, tagline, region, api_key):
    api_url = (
        f'https://'+ region + '.api.riotgames.com/riot/account/v1/accounts/by-riot-id/' 
        + summoner_name +'/' + tagline + '?api_key=' + api_key)
    print(f'Requesting API URL: {api_url}')
    ##API Request
    resp = requests.get(api_url)
    
    ##If statement for successful request otherwise prints status message
    if resp.status_code == 200:
        player_info = resp.json()
        puuid = player_info['puuid']
        return puuid
    else:
        print(f'Error:{resp.status_code}, {resp.text}')
        return None

#Calling function to get desired puuid
get_puuid('Phantóm', 'NA1', 'americas', api_key)
```
<br><br>
**Function #2 allows us to get the 20 recent Match IDS by passing PUUID and Region**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#This function returns 20 most recent matches by using the respective puuid and region
def get_matches(puuid, region, api_key):
    api_url = (f'https://' + region + '.api.riotgames.com/lol/match/v5/matches/by-puuid/' 
               + puuid + '/ids?start=0&count=20' + '&api_key=' + api_key)
    print(f'Requesting API URL: {api_url}')
    
    ##API Request
    resp = requests.get(api_url)

    ##If statement for successful request otherwise prints status message
    if resp.status_code == 200:
        match_ids = resp.json()
        return match_ids
    else:
        print(f'Error Received: {resp.status_code}', {resp.text})
        return None

#Calling function to get 20 most recent match IDS for specified user
get_matches('jq_A-Q1qsDMSLr7KbONxw_JBVPdGJq6hxcIgZq2GHynjwJyuKF45IGTAoFJmpPPo-36Fpm4qtl4BNQ', 'americas', api_key)
```
<br><br>
**Function #3 allows us to get the match data by passing Match ID and Region**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Function to get all match data by using match id and region parameters
def get_match_data(match_id, region, api_key):
    api_url = (f'https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}')
    
    ##API Request
    resp = requests.get(api_url)

    ##If statement for successful request otherwise prints status message
    if resp.status_code == 200:
        match_data = resp.json()
        return match_data
    else:
        print(f'Error Received: {resp.status_code}', {resp.text})
        return None
    
#Calling function to get all match data for a specified match id
get_match_data('NA1_5194605614', 'americas', api_key)
```
<br><br>
**Function #4 allows us to get the specific player data by passing Match Data and PUUID.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Retrieves the player data in JSON format by using match data and puuid.
#Intended to be used specifically for next function and not be called individually
def find_player_data(match_data, puuid):

    # Iterate over the participants in the match data
    for participant in match_data.get("info", {}).get("participants", []):
        if participant.get("puuid") == puuid:
            return participant  # Return the matching participant's data
    return None
```
<br><br>
**Function #5 is the final function, combining the previous two functions to retrieve player data. To obtain data for the last 20 matches, we use a for-loop to iterate through the list of match IDs, extracting the relevant data and storing it in a pandas DataFrame. This process returns a structured dataset with 20 rows, each representing a match, and columns defined within the function.**

<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Add dependency to convert data object into dataframe
import pandas as pd

##Cumulative function to get specific python list attributes for specified player data
def retrieve_all_data(puuid, region, api_key):
    
    ##Initializing dictionary called data so values can be appended
    data = {
        'champion': [],
        'kills': [],
        'deaths': [],
        'assists': [],
        'killing_sprees': [],
        'gold': [],
        'damage_dealt': [],
        'win': [],
        'Playtime (s)': []
    }

    ##Giving list of match ids for data to be extracted
    match_ids = ['NA1_5213549587',
'NA1_5213520491',
 'NA1_5212794969',
 'NA1_5212771073',
 'NA1_5210032454',
 'NA1_5210025036',
 'NA1_5210011121',
 'NA1_5209989228',
 'NA1_5207701701',
 'NA1_5207685142',
 'NA1_5205761116',
 'NA1_5205737558',
 'NA1_5194644149',
 'NA1_5194626272',
 'NA1_5194605614',
 'NA1_5187419555',
 'NA1_5187394878',
 'NA1_5187367363',
 'NA1_5187344821',
 'NA1_5186325149']

    ##For loop so below functions can run for each match id 
    for match_id in match_ids: 
        # run the two functions to get the player data from the match ID
        match_data = get_match_data(match_id, region, api_key)
        player_data = find_player_data(match_data, puuid)
        champion = (player_data['championName'])
        k = (player_data['kills'])
        a = (player_data['assists'])
        d = (player_data['deaths'])
        killing_sprees = (player_data['killingSprees'])
        gold = (player_data['goldEarned'])
        damage = (player_data['totalDamageDealt'])
        victory = (player_data['win'])
        playtime = (player_data['timePlayed'])

        ##Appending values to data dictionary object
        data['champion'].append(champion)
        data['kills'].append(k)
        data['deaths'].append(d)
        data['assists'].append(a)
        data['killing_sprees'].append(killing_sprees)
        data['gold'].append(gold)
        data['damage_dealt'].append(damage)
        data['win'].append(victory)
        data['Playtime (s)'].append(playtime)

    ##Converting data dictionary to dataframe object for analysis/manipulation
    df = pd.DataFrame(data)

    df['win'] = df['win'].astype(int) # change this column from boolean (True/False) to be integers (1/0)
    
    return df
```
<br><br>
## Data Transformation
Function is called to return DataFrame.
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Looking at df to understand table structure and data
df = retrieve_all_data('jq_A-Q1qsDMSLr7KbONxw_JBVPdGJq6hxcIgZq2GHynjwJyuKF45IGTAoFJmpPPo-36Fpm4qtl4BNQ', 'americas', api_key)

df
```

<img src="https://github.com/NikhilInampudi/LeagueOfLegends-Analysis/blob/de38b0ac20ebb70706426e4c3765406008def021/League%20Of%20Legends%20Dataframe.png" width="700" height="600" />

<br><br>
*Interesting Note:*
Form of Pre-Processing was done in the function when converting the Python dictionary into a Dataframe. By changing the values in the win column to 1/0 instead of win/loss, there is no need for label encoding later when attempting to fit the model. 
<div style="max-height: 400px; overflow-y: auto;">
    
```python
##Converting data dictionary to dataframe object for analysis/manipulation
    df = pd.DataFrame(data)

    df['win'] = df['win'].astype(int) # change this column from boolean (True/False) to be integers (1/0)
    
    return df
```
<br><br>
## Exploratory Data Analysis
Exploratory Data Analysis (EDA) is a critical step in the data analysis process. It involves investigating and summarizing the main characteristics of a dataset, often using visual methods, to understand its structure, patterns, and relationships before applying more formal statistical techniques or machine learning models. In this phase, I used different visualization techniques and methods to visualize correlations between variables, aggregation values by champion, and distribution of certain values. 

<br><br>
**Getting descriptive statistics about dataset**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Get descriptive statistics to understand data
df.describe()
```

<img src="https://github.com/NikhilInampudi/LeagueOfLegends-Analysis/blob/e2444fd3989c8ec63ad140b5fa74b9daa1ad6f0e/Match%20Statistics%20Output.png" width="900" height="400" />


<br><br>
**Manipulating pandas dataframe to get top champions by damage dealt**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Aggregating damage by champion and ordering by most to least
df_damage = df.groupby('champion', as_index=False)['damage_dealt'].sum()

df_damage = df_damage.sort_values(by='damage_dealt', ascending=False)

##Storing in new dataframe and getting top 10 rows
df_damage.head(10)
```

**Implementing matplotlib to visualize damage dealt per champion**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Adding dependencies for visualiziation creations
import matplotlib.pyplot as plt

##Creating bar chart to show damage by champion in descending order
color = ['lightcoral', 'gold', 'blue', 'orange', 'green', 'purple', 'orchid']

plt.figure(figsize = (8, 5))

plt.gcf().set_facecolor('darkgrey')
plt.gca().set_facecolor('black')

plt.bar(df_damage['champion'], df_damage['damage_dealt'], color=color)
plt.title('Damage by Champion', fontsize = 15, fontweight = 'bold')
plt.xlabel('Champions', fontweight = 'bold')
plt.ylabel('Damage Dealt', fontweight = 'bold')
plt.xticks(rotation=45)

plt.show()
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-Analysis/blob/1289aaa7dbbac301b55e4f44a6a13c9417413965/Visualizations/Damage%20by%20Champion.png" width="900" height="600" />


<br><br>
**Manipulating pandas dataframe to get top average deaths by champion**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Aggregating average deaths by champion 
df_deaths = df.groupby('champion', as_index=False)['deaths'].mean()

df_deaths = df_deaths.sort_values(by='deaths', ascending=False)

df_deaths.head()
```

**Creating lollipop chart to visualize champions which I average the most deaths**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Creating lollipop chart to visualize death average by champion played
plt.figure(figsize = (8, 5))

plt.gcf().set_facecolor('skyblue')

plt.stem(df_deaths['champion'], df_deaths['deaths'], linefmt='slategrey', markerfmt='indigo', basefmt=' ')
ax = plt.gca()
ax.set_facecolor('gainsboro')

plt.title('Deaths by Champion', fontsize=15, fontweight='bold')
plt.xlabel('Champions', fontweight='bold')
plt.ylabel('Average Deaths', fontweight='bold')
plt.xticks(rotation=45)

plt.show()
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-Analysis/blob/82a0843f668cb8dd6cf287e354a9136ef94aa180/Visualizations/Average%20Deaths%20by%20Champion%20Lollipop%20Chart.png" width="900" height="600" />



<br><br>
**Creating pie chart to visualize counts of win/loss for last 20 matches**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Creating pie chart to visualize win/loss split
win_loss = df.win.value_counts()

label = ['Wins', 'Losses']

plt.pie(win_loss, labels=win_loss)
plt.legend(label, title='Legend')
plt.title('Win/Loss', fontsize=15, fontweight='bold')

plt.show()
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-DataScience/blob/c73951b8a907496b1e835c080a0e24217964ac23/Visualizations/Win%20Loss%20Pie%20Chart.png" width="500" height="500" />


<br><br>
**Creating histogram to visualize killing spree distribution**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Creating histogram to visualize killing spree count distribution with measures of central tendency
killing_sprees = df.killing_sprees

mean_value = killing_sprees.mean()
median_value = killing_sprees.median()
min_value = killing_sprees.min()
max_value = killing_sprees.max()

plt.figure(figsize = (8, 5))

plt.hist(killing_sprees)

plt.axvline(x=mean_value, color='red')
plt.axvline(x=median_value, color='yellow')
plt.axvline(x=min_value, color='black')
plt.axvline(x=max_value, color='black')

plt.title('Killing Spree Distribution')
plt.xlabel('Killing Sprees')
plt.ylabel('Frequency')

plt.show()
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-DataScience/blob/921093d2a8fab62ebf8848acc48b17c571492579/Visualizations/Killing%20Spree%20Histogram.png" width="725" height="550" />


<br><br>
## Feature Engineering
Feature engineering is a crucial step in the machine learning pipeline that involves transforming raw data into meaningful features that can improve the performance of machine learning models.In this project, I applied feature engineering by creating new features with the goal of achieving a stronger correlation with the target variable, "win." My reasoning was that by enhancing these relationships, the model could gain better predictive power and more effectively uncover underlying patterns in the data.

<br><br>
**Creating new features called "Gold per minute" and "Damage dealt per minute"**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Creating new features to test stronger correlations
df['gold per Minute'] = round(df['gold'] / df['Playtime (s)'] * 60, 2)

df['damage_dealt per minute'] = round(df['damage_dealt'] / df['Playtime (s)'] * 60, 2)

df.head()
```

**Scaling features so there isn't any bias for values that are naturally higher. By doing this we can ensure that all the variables are treated fairly.** 
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Scaling features to standardize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[['gold per Minute', 'damage_dealt per minute']] = scaler.fit_transform(df[['gold per Minute', 'damage_dealt per minute']])

df[['kills', 'assists']] = scaler.fit_transform(df[['kills', 'assists']])

df.head()
```

<br><br>
**I utilized the seaborn library to generate a correlation heatmap, which provided a visual matrix illustrating the relationships between all variables. By analyzing this heatmap, I identified the variables with the strongest correlations to the target variable, "win." Although I engineered new features such as "Gold per minute" and "Damage dealt per minute," their correlation with "win" was weaker compared to the original variables. As a result, I opted to use "kills" and "assists" as the primary features for the model. Given the limited size of the dataset, I chose to keep the number of features minimal to avoid introducing unnecessary complexity, which could potentially hinder the model's performance.**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
import seaborn as sns

#Calculate the correlation matrix
df_corr = df.drop(columns=['champion'])
correlation_matrix = df_corr.corr()

##Step 2: Use seaborn to create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='inferno', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-DataScience/blob/0d570877faeb35faf36eee429e6f1cb2fd92c14f/Visualizations/Correlation%20Heatmap.png" width="775" height="600" />


<br><br>
## Machine Learning
I chose to implement a Logistic Regression algorithm due to its strong suitability for binary classification tasks, such as predicting wins and losses. Given the small size of the dataset, more complex models like neural networks or ensemble methods are often prone to overfitting. In contrast, Logistic Regression, as a simple and linear model, is less likely to overfit when working with limited data. Additionally, this algorithm performs exceptionally well with fewer features that exhibit a linear relationship with the target variable, making it an ideal choice for this specific problem.

<br><br>
**Implementing sci-kit learn library to create training / testing sets for our predictors / target variables.** 
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Adding dependencies for train/test, model building, and evluation metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

##Creating x variable for features and y variable for target
x = df.drop(columns = ['win', 'gold', 'killing_sprees', 'damage_dealt', 'champion', 'deaths', 'Playtime (s)', 'damage_dealt per minute', 'gold per Minute'])
y = df['win']

##Splitting data into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=42)
```
**Creating the model and fitting 80% of our dataset as training data**   
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Creating Logistic Regression Model 
model = LogisticRegression()

#Fitting training features and target into model
model.fit(x_train, y_train)
```
**Using the model to predict the testing features**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Make predictions based off test features
prediction = model.predict(x_test)
```
**Using different evluation metrics to compare the results of the actuals in our testing set and the prediction**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Evaluation methods to compare output accuracy
accuracy = accuracy_score(y_test, prediction)
print('Accuracy Score:')
print(accuracy)

matrix = confusion_matrix(y_test, prediction)
print('Confusion Matrix Score:')
print(matrix)

report = classification_report(y_test, prediction)
print('Classification Report')
print(report)
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-DataScience/blob/dcd56d03ce593251173eae4803f80c2e8fb94a0f/Evaluation%20Output.png" width="700" height="400" />
<br><br>

**Predicting outcome of match with new data**
<div style="max-height: 400px; overflow-y: auto;">
    
```python
#Testing model on newly created dataset to get win/losses
new_data = [[27, 20], [10, 30], [4, 20], [18, 9], [20, 10], [25, 18]]

new_data = scaler.transform(new_data)

new_prediction = model.predict(new_data)

print(f'New Predictions: {new_prediction}')
```
<img src="https://github.com/NikhilInampudi/LeagueOfLegends-DataScience/blob/0da6250578e393c6223a141eecb31d2f7ed82eb9/Prediction%20Output.png" width="500" height="50" />


## Results and Insights
- **Kills and Assists**: These were the most significant predictors of match outcomes, with higher values strongly correlating with wins.
- **Champion Performance**:
     - Certain champions consistently outperformed others in terms of damage dealt such as Twitch, Ezreal, and Jinx.
     - Certain champions I am more prone to dying on such as Varus, Lux, LeBlanc.
     - I get an average of 4.5 - 5 killing sprees a match which means I am more likely to go on long kill streaks.  
- **Win/Loss Ratio**: The model achieved an accuracy of 75% in predicting match outcomes based on total-game statistics. 













