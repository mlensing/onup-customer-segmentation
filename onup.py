# OnUp Segmentation

import os
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import chart_studio as py
import plotly.figure_factory as ff
import plotly 
%matplotlib inline
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise

# Make sure to save the "Condensed_OnUp_Survey_Results.csv" to a folder on your computer 
# Then change the file path below
os.chdir("/Users/michellelensing/Desktop/Senior_Year/Marketing_Analytics/OnUp_Project")
onup_data = pd.read_csv('Condensed_OnUp_Survey_Results.csv', encoding='ISO-8859-1')

onup_data = onup_data.loc[:, ~onup_data.columns.str.contains('^Unnamed')]
onup_data.head()



# Cleaning the Data

# Renames columns, shorter names
onup = pd.DataFrame(onup_data)
onup_data = onup.rename(columns={"How often do you complete a cardio work out?": "CardioFreq",
                     "How often do you complete a weight/resistance training work out?": "StrengthFreq",
                    "What type of athlete would you consider yourself to be?": "AthleteType",
                    "What is your age?": "Age", "What is your gender?": "Gender",
                    "Thinking of your cardio workouts, on a scale of 1-5, how interested are you in trying a protein water for a cardio workout?":
                    "CardioInterest", "On a scale of 1-5, how interested are you in trying a protein water for a resistance / weight workout?":
                    "StrengthInterest", "How many athletic competitions have you paid a registration fee to participate in over the past 2 years?": "NumCompetitions",
                                "If you had to choose one word to describe your current athletic habits, what would it be?": "AthleticHabits",
                                "How would you describe yourself?": "FitnessView",
                                "How often do you use protein in conjunction with your cardio workouts? (pre, during, or post)": "CardioProtein",
                                "How often do you use protein in conjunction with your weight / resistance workouts? (pre, during, or post)": "StrengthProtein",
                                "What do you know about protein water?": "Knowledge",
                                "How many ounces would you like your protein water to be?": "Ounces",
                                "How much protein would you want in a protein water (in grams)?": "ProteinLevel",
                                "How many calories would you prefer a protein water to be?": "Calories",
                                "How would you prefer your protein water to be sweetened?": "Sweetened",
                                "Are you familiar with leucine?": "LeucineInfo", "Whey protein is made up of amino acids including leucine. Leucine is particularly important because it triggers muscle repair. If you knew that you needed to consume 25g of whey protein following a workout to trigger muscle repair, what would you do?":
                                "LeucineLearn", "What characteristic would be most important to you in a protein water? (rank order) [Protein Level]": "Protein_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [Organic]": "Organic_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [Taste]": "Taste_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [All Natural]": "Natural_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [Refreshing]": "Refreshing_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [Less than 100 calories]": "Under100Cal_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [No sugar]": "NoSugar_Imp",
                                "What characteristic would be most important to you in a protein water? (rank order) [With electrolytes]": "Electrolytes_Imp"})
onup_data.head()

def age_groups(series):
    # recodes age group data in groups from 1 to 7
    if series == "<20":
        return "1"
    elif series == "21-27":
        return "2"
    elif series == "28-33":
        return "3"
    elif series == "34-40":
        return "4"
    elif series == "41-47":
        return "5"
    elif series == "48-54":
        return "6"
    elif series == "55+":
        return "7"
    
onup_data['Age'] = onup_data['Age'].apply(age_groups)

def cardio_freq(series):
    # recodes cardio frequency into variables 0-3
    if series == "1-2 days per week for at least 30 minutes":
        return "1"
    if series == "3-4 days per week for at least 30 minutes":
        return "2"
    if series == "5+ days per week for at least 45 minutes":
        return "3"
    
onup_data['CardioFreq'] = onup_data['CardioFreq'].apply(cardio_freq)


def strength_freq(series):
    # recodes strength frequency into variables 0-3
    if series == "1-2 days per week":
        return "1"
    if series == "3-4 days per week":
        return "2"
    if series == "5+ days per week":
        return "3"
    
onup_data['StrengthFreq'] = onup_data['StrengthFreq'].apply(strength_freq)

def athlete_type(series):
    # recodes type of athlete into variables 1-3
    if series == "Cardio / Endurance":
        return "1"
    if series == "Both":
        return "2"
    if series == "Weight / Resistance":
        return "3"
    
    
onup_data['AthleteType'] = onup_data['AthleteType'].apply(athlete_type)

def gender(series):
    # recodes female to 1 and male to 2
    if series == "Female":
        return "1"
    if series == "Male":
        return "2"
  
onup_data['Gender'] = onup_data['Gender'].apply(gender)

def num_competitions(series):
    # recodes number of athletic competitions into groups 
    if series == "0":
        return "0"
    if series == "1 to 3":
        return "1"
    if series == "4 to 6":
        return "2"
    if series == "7 to 9":
        return "3"
    if series == "10+":
        return "4"
  
onup_data['NumCompetitions'] = onup_data['NumCompetitions'].apply(num_competitions)

def cardio_protein(series):
    # recodes how often respondents use protein with cardio workouts into groupings 
    if series == "I donÕt use protein in conjunction with my cardio workouts":
        return "0"
    if series == "25% of the time":
        return "1"
    if series == "50% of the time":
        return "2"
    if series == "75% of the time":
        return "3"
    if series == "100% of the time":
        return "4"
  
onup_data['CardioProtein'] = onup_data['CardioProtein'].apply(cardio_protein)

def strength_protein(series):
    # recodes how often respondents use protein with strength workouts into groupings 
    if series == "I donÕt use protein in conjunction with my cardio workouts":
        return "0"
    if series == "25% of the time":
        return "1"
    if series == "50% of the time":
        return "2"
    if series == "75% of the time":
        return "3"
    if series == "100% of the time":
        return "4"
  
onup_data['StrengthProtein'] = onup_data['StrengthProtein'].apply(strength_protein)

def fitness_view(series):
    # recodes fitness views into variables 1-3 from least "obsessed" to most "obsessed"
    if series == "Fitness is a take it or leave it type of thing":
        return "1"
    if series == "I love fitness, but life often gets in the way so I dont workout as much as Id like":
        return "2"
    if series == "I prioritize fitness over other activities":
        return "3"
    
    
onup_data['FitnessView'] = onup_data['FitnessView'].apply(fitness_view)

def protein_water_knowledge(series):
    # recodes knowledge on protein water into variables 0-3 from never heard of it to drink it on a regular basis
    if series == "Never heard of it":
        return "0"
    if series == "Heard of it, but have never purchased" or series == "Heard of it, but have never seen it at a store":
        return "1"
    if series == "Purchased in the past, but not a regular user":
        return "2"
    if series == "I drink it on a regular basis":
        return "3"
    
    
onup_data['Knowledge'] = onup_data['Knowledge'].apply(protein_water_knowledge)

def calorie_pref(series):
    # recodes preference for calorie amount into variables 0-3 from least to most
    if series == "Less than 50 calories":
        return "1"
    if series == "Less than 100 calories":
        return "2"
    if series == "100 calories":
        return "3"
    if series == "100-200 calories":
        return "4"
    if series == "200+ calories":
        return "5"
    
    
onup_data['Calories'] = onup_data['Calories'].apply(calorie_pref)

onup_data.head()

onup_data_updated = onup_data[['CardioFreq', 'StrengthFreq', 'AthleteType', 'Age', 'Gender', 'CardioInterest', 'StrengthInterest']]

# Fill null values with "0"
onup_data_updated = onup_data_updated.fillna(0)
onup_data = onup_data.fillna(0)

onup_data_updated.head()

# Taking a Look at Hierarchical Clustering

# Looking at cluster assignments using hiearchical clustering first
avg_link = hierarchy.linkage(onup_data_updated, 'average')
dn7 = hierarchy.dendrogram(avg_link, above_threshold_color='#bcbddc', orientation='right',
                           truncate_mode='lastp', p=100,
                           leaf_rotation=90.,
                           leaf_font_size=10.,
                           show_contracted=True)

# K-Means Clustering

# k-means method
kmeans = KMeans(n_clusters=4, random_state=1).fit(onup_data_updated)

# look at the cluster membership for each item in dataset 
kmeans.labels_

# cluster centers, means for each variable and cluster
pd.DataFrame(data=kmeans.cluster_centers_)

# These are the variables that the k-means segmenting was based on
#See top of code for original questions to fully understand each variable
onup_data_updated.head()

# add cluster membership to data set
k_clusters_4 = pd.DataFrame(data=kmeans.labels_[0:])
onup_data['k_clusters_4']= k_clusters_4
onup_data.head()

# Profiling Segments

# update data type to integer for all data

onup_data_updated = onup_data_updated.astype(int)

onup_data_new = onup_data[['CardioFreq', 'StrengthFreq', 'AthleteType', 'Age', 'Gender', 'CardioInterest', 'StrengthInterest', 'NumCompetitions', 'CardioProtein','StrengthProtein', 'FitnessView', 'k_clusters_4']]

# update data type to integer for all data
onup_data_new = onup_data_new.astype(int)

import statsmodels.api as sm
from statsmodels.formula.api import ols

#import the library to help us table the means
import researchpy as rp
#create a summary of the means
rp.summary_cont(onup_data_new['NumCompetitions'].groupby(onup_data_new['k_clusters_4']))
#this looks at how many competitions respondents in each cluster did in the past year

# compares how often respondents in each cluster already use protein in conjuction with cardio workouts
rp.summary_cont(onup_data_new['CardioProtein'].groupby(onup_data_new['k_clusters_4']))

# compares how often respondents in each cluster already use protein in conjuction with strength workouts
rp.summary_cont(onup_data_new['StrengthProtein'].groupby(onup_data_new['k_clusters_4']))

# compares each clusters fitness view mean (going from least fitness obsessed to most fitness obsessed)
rp.summary_cont(onup_data_new['FitnessView'].groupby(onup_data_new['k_clusters_4']))

# create the ANOVA model
all_comp_values = sorted(onup_data_new["k_clusters_4"].unique())
mod = ols('NumCompetitions ~ C(k_clusters_4, levels = all_comp_values)', data = onup_data_new).fit()
mod.summary()

#post hoc tests for pairwise comparison of categories from the ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(onup_data_new['NumCompetitions'], onup_data_new['k_clusters_4'])
mc_results = mc.tukeyhsd()
print(mc_results)

onup_data_new.head()

# Comparing Strength, Cardio, and Both Groups

onup_preferences = onup_data[['AthleteType', 'Knowledge', 'ProteinLevel', 'Calories', 'Protein_Imp', 'Organic_Imp', 'Taste_Imp',
                             'Natural_Imp', 'Refreshing_Imp', 'Under100Cal_Imp', 'NoSugar_Imp', 'Electrolytes_Imp']]
onup_preferences = onup_preferences.astype(int)
onup_preferences.head()

# compares each athlete type's knowledge means
# 0 - never heard of it
# 1 - heard of it, but have never purchased OR heard of it, but have never seen it at a store
# 2 - purchased in the past, but not a regular user
# 3 - I drink it on a regular basis
rp.summary_cont(onup_preferences['Knowledge'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's desired protein level (in grams) means
rp.summary_cont(onup_preferences['ProteinLevel'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's desired calorie level means
# 1 - Less than 50 calories
# 2 - Less than 100 calories
# 3 - 100 calories
# 4 - 100-200 calories
# 5 - 200+ calories
rp.summary_cont(onup_preferences['Calories'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for protein means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Protein_Imp'].groupby(onup_data_new['AthleteType']))


# compares each athlete type's level of importance for organic means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Organic_Imp'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for taste means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Taste_Imp'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for natural means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Natural_Imp'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for refreshing means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Refreshing_Imp'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for under 100 calories means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Under100Cal_Imp'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for no sugar means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['NoSugar_Imp'].groupby(onup_data_new['AthleteType']))

# compares each athlete type's level of importance for electrolytes means (1 is most, 8 is least)
rp.summary_cont(onup_preferences['Electrolytes_Imp'].groupby(onup_data_new['AthleteType']))

onup_data.head()

strength_group = onup_data.loc[onup_data['AthleteType'] == '3']
cardio_group = onup_data.loc[onup_data['AthleteType'] == '1']
cardio_and_strength_group = onup_data.loc[onup_data['AthleteType'] == '2']

onup_data['AthleticHabits'].unique()

def num_of_each_AthleticHabit(athlete_type_df, athlete_type_chosen):
    print('Number of Each Athletic Habit for', athlete_type_chosen)
    print('Cross Trainer:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Cross Trainer']))
    print('Team Participant:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Team Participant (tennis, basketball, baseball, soccer, etc)']))
    print('Runner:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Runner']))
    print('Crossfitter:', len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Crossfitter']))
    print('Cyclist:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Cyclist']))
    print('Weight Trainer:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Weight Trainer']))
    print('Triathlete:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Triathlete']))
    print('Swimmer:',len(athlete_type_df.loc[athlete_type_df['AthleticHabits'] == 'Swimmer']))


num_of_each_AthleticHabit(strength_group, 'Strength')

num_of_each_AthleticHabit(cardio_group, 'Cardio')

num_of_each_AthleticHabit(cardio_and_strength_group, 'Both')

