import pandas as pd

# Load the dataset
df = pd.read_csv('/content/car_prices.csv')

# Clean the dataset by identifying and eliminating rows with missing or NULL values
car_prices_clean = df.dropna().copy() # We use .copy() to avoid Setting With Copy Warning, This happens because pandas isn't sure whether the modification is being made to a copy of the data or a view on the original data, leading to potential confusion or unexpected behavior in your data manipulation

# Display the shape of the original vs cleaned dataset to see how many rows had missing values
original_shape = df.shape
cleaned_shape = car_prices_clean.shape

original_shape, cleaned_shape

# Check for duplicate rows
duplicates = car_prices_clean.duplicated().sum()

# Check data types to ensure they are appropriate for each column
data_types = car_prices_clean.dtypes

# Convert 'saledate' to datetime format for better handling of dates
car_prices_clean.loc[:, 'saledate'] = pd.to_datetime(car_prices_clean['saledate'], errors='coerce') #This will give a  Deprecation Warning, which means that when you use .loc[:, 'saledate'] to assign new values to the 'saledate' column, pandas is warning that the way these in-place modifications work will change in a future version. This warning aims to alert you to potential changes in your code's behavior with future pandas updates.

# Re-check data types after conversion
updated_data_types = car_prices_clean.dtypes
duplicates, data_types, updated_data_types

## EDA
# Preliminary Data Analysis
dat.head()
dat.info()
dat.describe()
dat.count()
dat['mmr'].skew()
dat['mmr'].kurt()


# Setting the visual style
sns.set(style="whitegrid")

# Plotting the distribution of Market Values
plt.figure(figsize=(8, 6))
sns.histplot(dat['mmr'], bins=100, kde=True, palette='viridis')
plt.title('Distribution of Market Values')
plt.xlabel('Market Values')
plt.ylabel('Frequency')

# Set x-axis limit up to $75,000
plt.xlim(left=0, right=75000)
plt.show()

# 1. Plotting the Mileage vs. MMR

plt.figure(figsize=(10, 6))
sns.scatterplot(data=dat, x='odometer', y='mmr', palette='viridis')
plt.title('Odometer vs. MMR')
plt.xlabel('Odometer')
plt.ylabel('MMR')
plt.show()

# 2. Plotting the Age of the Car (Year) vs. Market Value (MMR)

# Calculating the car's age by subtracting the car's year from the current year.
# Assuming the current year for this dataset's context; use 2024 for demonstration purposes
dat['car_age'] = 2024 - dat['year']

# Creating a line plot for the average market value (MMR) by the age of the car
plt.figure(figsize=(10, 8))
sns.lineplot(data=dat, x='car_age', y='mmr', estimator='mean', ci=None, palette='viridis') # means that for each unique value of car_age, the mean mmr is calculated and plotted #ci confidence interval
plt.title('Average Market Value vs. Age of the Car')
plt.xlabel('Age of the Car (Years)')
plt.ylabel('Average Market Value (MMR)')
plt.show()

# 3. Plotting Market Value vs. Brand

plt.figure(figsize=(12, 12))
# Assuming 'brand (Car make)' is a categorical variable and 'market_value' is continuous
average_market_value_by_brand = dat.groupby('make')['mmr'].mean().reset_index()
average_market_value_by_brand = average_market_value_by_brand.sort_values('mmr', ascending=False)
sns.barplot(data=average_market_value_by_brand, x='mmr', y='make', palette='viridis')
plt.title('Average Market Value by Brand')
plt.xlabel('Average Market Value')
plt.ylabel('Brand')
plt.show()

# 4. Plotting the Condition of the Car vs. the Market Value

# Ensure the 'condition' column is numeric and can be grouped
dat['condition'] = pd.to_numeric(dat['condition'], errors='coerce')
# If an error is encountered, all invalid parsing will be set as NaN (Not a Number).
#This option is useful when you are okay with losing some data in exchange for converting as much of the data to a numeric type as possible without interruption.

# Creating a scatter plot for the average MMR by the new (corrected) condition groups
plt.figure(figsize=(10, 8))
sns.scatterplot(data=dat, x='condition', y='mmr')
plt.title('MMR by Car Condition')
plt.xlabel('Car Condition')
plt.ylabel('Market Value (MMR)')
plt.show()

# Calculating the correlation matrix
correlation_matrix = dat.corr()

# Creating a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# The use of fmt=".2f" alongside annot=True ensures that the correlation values in the heatmap are displayed with two decimal places
plt.title('Correlation Matrix of Dataset Variables')
plt.show()

# 5. Plotting Car age vs. Mileage since we find a higher correlation of 0.77 in the above correlation matrix

plt.figure(figsize=(12, 8))
sns.scatterplot(x=dat['car_age'], y=dat['odometer'])
plt.title('Car Age vs. Odometer Reading')
plt.xlabel('Car Age (Years)')
plt.ylabel('Odometer Reading (Miles)')
plt.show()

# Using a t-test on car_age and MMR
# Using the median car age as the cutoff
median_age = dat['car_age'].median()
median_age

# Create two groups based on the median age
newer_cars = dat[dat['car_age'] <= median_age]['mmr'].dropna()
older_cars = dat[dat['car_age'] > median_age]['mmr'].dropna()

from scipy.stats import ttest_ind

# Perform t-test
t_stat, p_val = ttest_ind(newer_cars, older_cars, equal_var=False)  # Assuming unequal variances

print(f"T-statistic: {t_stat}, P-value: {p_val}")

# Correlation Analysis for Categorical and Continuous Variables

!pip install dython #dython library
from dython.nominal import correlation_ratio #import function

# Iterate over your categorical variables and calculate the correlation ratio with the continuous variable 'MMR'
correlation_ratios = {} # is a dictionary with key, value pair. Key: categorical var & value: corr ratio
# Initialize an empty dictionary

for column in dat.select_dtypes(include=['object']).columns: # for loop, In pandas, cat var. is often rep as "objects"
    if column != 'mmr':  # ensure that the current column in the iteration is not 'mmr'
        correlation_ratios[column] = correlation_ratio(dat[column], dat['mmr']) # column stores categorical var as keys

# Convert the dictionary to a sorted list to view the categorical variables by their correlation ratio
sorted_correlation_ratios = sorted(correlation_ratios.items(), key=lambda item: item[1], reverse=True)
#key=lambda: lambda function used to extract a comparison key from each list element during sorting.
# item[1]: ensuring that the sorting is done according to the strength of the correlation.

# Print out the sorted correlation ratios
for column, ratio in sorted_correlation_ratios:
    print(f"Categorical Variable: {column}, Correlation Ratio with MMR: {ratio}")

# Convert the sorted list of tuples to a DataFrame for easier plotting
correlation_df = pd.DataFrame(sorted_correlation_ratios, columns=['Categorical Variable', 'Correlation Ratio'])

# Set the figure size
plt.figure(figsize=(15, len(correlation_df) * 0.5))

# Create the heatmap
sns.heatmap(correlation_df.set_index('Categorical Variable').T, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation Ratio'})
#annot: to add text annotations on each cell of the heatmap
#cbar_kws: to add a color bar to specifying what the color intensities represent

# Set the title and adjust layout
plt.title('Correlation Ratios of Categorical Variables with MMR')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# 1. MMR vs. Model
# Filter to include only top models for clarity in visualization
top_models = dat['model'].value_counts().nlargest(10).index
filtered_dat = dat[dat['model'].isin(top_models)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='model', y='mmr', data=filtered_dat)
plt.xticks(rotation=45)
plt.title('MMR Distribution by Top Models')
plt.xlabel('Model')
plt.ylabel('MMR')
plt.show()

# 2. MMR vs. Seller
top_sellers = dat['seller'].value_counts().nlargest(10).index
filtered_dat_seller = dat[dat['seller'].isin(top_sellers)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='seller', y='mmr', data=filtered_dat_seller)
plt.xticks(rotation=45)
plt.title('MMR Distribution by Top Sellers')
plt.xlabel('Seller')
plt.ylabel('MMR')
plt.show()

# 3. MMR vs. Trim
top_trims = dat['trim'].value_counts().nlargest(10).index
filtered_dat_trim = dat[dat['trim'].isin(top_trims)]

plt.figure(figsize=(12, 6))
sns.stripplot(x='trim', y='mmr', data=filtered_dat_trim, jitter=True) #jitter:to prevent overlapping and get visual clarity
# jitter: specially for cat vars, where pts can overlap
plt.xticks(rotation=50)
plt.title('MMR Distribution by Top Trims')
plt.xlabel('Trim')
plt.ylabel('MMR')
plt.show()

