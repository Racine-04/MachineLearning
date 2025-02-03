import great_expectations as gx
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load dataset from hugging face
data = load_dataset("GihanPramod99/House_Price")

# Check how the data is organized
print(f"{data}\n")

# Convert the data into a dataframe
df = pd.DataFrame(data["train"])

#See the info of the df
print(f"{df.head()}\n")
print(f"{df.info()}\n")

# Groups the df by city, counts the occurrences of each city, and creates a bar plot
city_grouping = df.groupby("city").size()
city_grouping.plot(kind = "bar")
plt.title("Frequency of Cities in the Dataset")
plt.show()

# Groups the df by country, counts the occurrences of each country, and creates a bar plot
country_grouping = df.groupby("country").size()
country_grouping.plot(kind = "bar")
plt.title("Frequency of Country in the Dataset")
plt.show()

# Groups the df by the year built, counts the occurrences of each year, and creates a bar plot
yr_built_grouping = df.groupby("yr_built").size()
yr_built_grouping.plot(kind = "bar")
plt.title("Frequency of the Year in the Dataset")
plt.show()

# Print description of the price
print(df["price"].describe())

# Plots the distrubititons of the price using a histogram of every value
df["price"].hist(bins=500)
plt.title("Histogram of Price")
plt.show()

# Plots the distrubutions of the price of a certain range based of the  info in the description like mean, std and percentiles 
plt.hist(df["price"], bins =50, range=[1e+03, 2e+06])
plt.title("Histogram of Price")
plt.show()

# Plots a correlation matrix of the numerical features in the dataset
correlation_matrix = df.corr(numeric_only=True)
plt.matshow(correlation_matrix)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.colorbar()
plt.show()

# Following this tutorial https://docs.greatexpectations.io/docs/core/introduction/try_gx/ to implement gx

# Creates a data context to manage expectations and validations
context = gx.get_context()

# Connects the pandas DataFrame to the data context and creates a batch to validate
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

# Defines the batch, specifying that the whole DataFrame should be validated
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

# Create various expectations for different columns in the dataset

# Assumes that the minimum price for a house should be above 50,000, especially in the current market
expectation1 = gx.expectations.ExpectColumnValuesToBeBetween(column="price", min_value=50000)

# Assumes that street addresses should be unique (no duplicates or inconsistencies)
expectation2 = gx.expectations.ExpectColumnValuesToBeUnique(column="street")

# Ensures that statezip values start with "WA " followed by five consecutive digits (e.g., "WA 98101")
expectation3 = gx.expectations.ExpectColumnValuesToMatchRegex(column="statezip", regex="WA [0-9][0-9][0-9][0-9][0-9]")

# Assumes that the size of the lot (sqft_lot) should always be greater than 0, since a house cannot have no lot
expectation4 = gx.expectations.ExpectColumnValuesToBeBetween(column="sqft_lot", min_value=1)

# Ensures that the living size (sqft_living) should always be larger or equal than the above area size (sqft_above)
expectation5 = gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A="sqft_living", column_B="sqft_above", or_equal=True)

# Ensures that a house has at least one floor (no houses with zero floors)
expectation6 = gx.expectations.ExpectColumnValuesToBeBetween(column="floors", min_value=1)

# Checks that the year a house was renovated should be after the year it was built; if not renovated, both years should be the same
expectation7 = gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A="yr_renovated", column_B="yr_built", or_equal=True)

# Assumes that the house condition rating should always be between 1 and 5
expectation8 = gx.expectations.ExpectColumnValuesToBeBetween(column="condition", min_value=1, max_value=5)

# Ensures that a house must have at least one bedroom
expectation9 = gx.expectations.ExpectColumnValuesToBeBetween(column="bedrooms", min_value=1)

# Ensures that a house must have at least one hlaf bathroom
expectation10 = gx.expectations.ExpectColumnValuesToBeBetween(column="bathrooms", min_value=0.5)

# Prints the validation result for the batch
for expectation in [expectation1, expectation2, expectation3, expectation4, expectation5, expectation6, expectation7, expectation8, expectation9, expectation10]:
    validation_result = batch.validate(expectation)
    print(f"{validation_result}\n")