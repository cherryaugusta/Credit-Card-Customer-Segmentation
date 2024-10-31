# Credit Card Customer Segmentation: An Unsupervised Machine Learning Analysis

This project is based on Dataquest’s guided project, ["Credit Card Customer Segmentation,"](https://www.dataquest.io/projects/guided-project-a-credit-card-customer-segmentation/) which provided an initial framework and dataset. I completed this project as part of my own learning experience.

## Project Introduction

In this project, the role of a data scientist at a credit card company is assumed. The task involves analyzing a dataset containing information about the company’s clients, with the aim of segmenting them into distinct groups. This segmentation will allow the company to tailor different business strategies to each customer group.

The company expects a segment for every client, accompanied by a detailed explanation of each group’s characteristics and the key factors that differentiate them.

During a planning session with the Data Science coordinator, it was decided to use the K-means algorithm for the segmentation process.

To effectively apply this algorithm and meet the company’s objectives, the following steps will be taken:

- Analyze the dataset;
- Prepare the data for modeling;
- Determine the optimal number of clusters;
- Perform the segmentation;
- Interpret and explain the results.

**Key Tools**: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)

## Exploratory Data Analysis

### **Dataset Overview**

The dataset consists of 10,127 records with 14 features, each providing information about customer demographics and financial behavior:

- **customer_id**: Unique identifier for each customer.
- **age**: Customer's age.
- **gender**: Customer's gender.
- **dependent_count**: Number of dependents.
- **education_level**: Customer's education level.
- **marital_status**: Customer's marital status.
- **estimated_income**: Estimated annual income.
- **months_on_book**: Duration of the customer’s relationship with the company in months.
- **total_relationship_count**: Number of relationships the customer has with the company.
- **months_inactive_12_mon**: Number of months the customer has been inactive in the last 12 months.
- **credit_limit**: Credit limit assigned to the customer.
- **total_trans_amount**: Total transaction amount.
- **total_trans_count**: Total number of transactions.
- **avg_utilization_ratio**: Average credit utilization ratio.

### **Data Integrity**

There are no missing values in the dataset, ensuring completeness and readiness for analysis.

### **Correlation Analysis**

The correlation matrix reveals several notable relationships:

1. **Age**:
   - Positively correlates with **months_on_book** (0.79), indicating that older customers tend to maintain longer relationships with the company.
   - Shows weak correlations with financial metrics such as **credit_limit** (0.002) and **total_trans_amount** (-0.046).

2. **Estimated Income**:
   - Strongly correlates with **credit_limit** (0.516), suggesting that higher-income customers are assigned higher credit limits.
   - Negatively correlates with **avg_utilization_ratio** (-0.278), implying that wealthier customers tend to use a lower proportion of their credit.

3. **Credit Limit**:
   - Negatively correlates with **avg_utilization_ratio** (-0.483), suggesting that customers with higher credit limits tend to utilize a smaller proportion of their available credit.
   - Moderately correlates with **total_trans_amount** (0.172), indicating that higher credit limits might be associated with greater spending.

4. **Total Transactions**:
   - **total_trans_amount** and **total_trans_count** are highly correlated (0.807), implying that more transactions lead to higher total spending.

5. **Other Observations**:
   - **Total_relationship_count** negatively correlates with both **total_trans_amount** (-0.347) and **total_trans_count** (-0.242), indicating that customers with fewer relationships may spend less and perform fewer transactions.
   - **months_inactive_12_mon** shows weak correlations with most variables, suggesting that inactivity does not strongly impact other financial behaviors.

### **Summary**

The data indicates that:
- Older customers are more likely to have long-term relationships with the company.
- Higher-income customers tend to have higher credit limits and lower credit utilization.
- More transactions generally lead to higher transaction amounts, but this does not significantly impact credit utilization.

These insights can inform strategies such as designing personalized financial products or improving customer retention by targeting high-utilization customers.

## Feature Engineering

The three categorical variables in the dataset, **gender**, **education_level**, and **marital_status**, require transformation for machine learning models.

- **gender** is converted into binary values (e.g., 0 and 1).
- **education_level** is transformed into numerical categories using the `replace()` method.
- **marital_status** is one-hot encoded, creating binary indicators for each status (e.g., **Married**, **Single**, **Unknown**).

### Feature Engineering Analysis

The feature engineering process prepares the data for further modeling by ensuring that categorical variables are appropriately encoded. Additional preprocessing such as normalizing continuous features can enhance the model's performance by ensuring comparability across variables.

## Scaling the Data

Scaling is essential to ensure that features are on a comparable scale, especially for algorithms like K-means that rely on distance calculations.

1. **Age**:
   - Scaled values range from -0.79 to 0.58, where values near 0 represent the average age.

2. **Gender**:
   - Encoded and standardized, gender values range from -0.94 to 1.06.

3. **Dependent Count**:
   - Values range from 0.50 to 2.04, indicating higher-than-average dependent counts for values above 0.

4. **Education Level**:
   - Values range from -1.46 to 0.66, where positive values may represent higher education levels.

5. **Estimated Income**:
   - Values range from -0.97 to 0.79, with positive values indicating higher-than-average incomes.

6. **Credit Limit** and **Total Transactions**:
   - Both are standardized, ensuring they contribute equally in clustering and other analyses.

### Summary

The scaling process helps improve model performance by ensuring that all features contribute equally, particularly in distance-based models like K-means.

## Choosing K

The optimal number of clusters was determined using the elbow method, which revealed that 6 clusters best balanced the trade-off between inertia and the number of clusters. A **CLUSTER** column was added to the dataset to represent the cluster assignments.

## Cluster Analysis

1. **Demographic Insights**:
   - The customer base is predominantly middle-aged, with notable variability in income and spending habits across clusters.

2. **Financial Behavior**:
   - Customers display a broad range of incomes, credit limits, and transaction patterns, from high spenders to financially constrained individuals.

3. **Cluster Insights**:
   - Cluster characteristics vary, with some clusters representing higher-spending, low-utilization customers, while others represent financially constrained, high-utilization customers.

### Cluster Distribution

Clusters are unevenly distributed, with **Cluster 4** containing the highest number of customers (2,781), followed by **Cluster 3** (2,490). Smaller clusters like **Cluster 6** (734) may represent niche customer segments requiring closer investigation.

### Recommendations for Business Strategy

- **Targeted Marketing**: High-income clusters can be offered premium services, while clusters with higher utilization may benefit from credit limit adjustments.
- **Churn Management**: Monitoring clusters with long periods of inactivity can help identify at-risk customers, leading to more effective re-engagement strategies.
  
## Key Cluster Summaries

### Cluster 1
- **Characteristics**: High credit limits, substantial spending, low credit utilization.
- **Opportunity**: Encourage increased card usage.

### Cluster 2
- **Characteristics**: Low income, high utilization, predominantly married.
- **Opportunity**: Focus on offering credit limit increases or budget management tools.

### Cluster 3
- **Characteristics**: Low credit limits and high utilization.
- **Opportunity**: Provide financial education to help manage credit usage.

### Cluster 4
- **Characteristics**: High-income, low credit utilization.
- **Opportunity**: Incentivize higher spending with personalized offers.

### Cluster 5
- **Characteristics**: Older, long-term customers with high utilization.
- **Opportunity**: Offer rewards for loyalty and encourage more frequent transactions.

### Cluster 6
- **Characteristics**: Single individuals with low income and high credit utilization.
- **Opportunity**: Adjust credit limits to better match customer needs and behaviors.

### Conclusion

The segmentation of the customer base reveals distinct groups with varying behaviors and financial needs. Understanding these clusters allows for more personalized marketing and customer service strategies, ultimately leading to better customer satisfaction and business performance.
