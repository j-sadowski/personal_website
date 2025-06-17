---
layout: post
title: How to analyze stratified random sampled data
date: 2019-12-12 10:00:00 -0500
categories: [data-science]
author: Jason Sadowski
---
Note: This post was originally published on [medium](https://medium.com/data-science/how-to-analyze-stratified-random-sampled-data-e3933199ae74).


When constructing an experiment, one of the most important questions to ask is: How should I sample from my population? Clearly you want to conduct some sort of random sampling procedure, but exactly how the random sampling is done can have a large impact on your analysis. While there are many useful guides on how to conduct [stratified random sampling](https://en.wikipedia.org/wiki/Stratified_sampling), I’ve noticed that there are few guidelines on how to correctly analyze your stratified sampled data. In this article, I’m going to discuss how to conduct stratified sampling and how to analyze the resultant data using some simulated data as an example.  

## Designing the experiment

First, I want to briefly go over stratified sampling and why it is important. Let’s say I am the data scientist at Company X. Company X has user base of about 500,000 people from across the United States. Most of our users live in cities, but there is also a large proportion that live in small towns. I’m interested in running an experiment to see how changing the user interface of our app could change the click-through probability from the app onto a secondary web site. A pretty straightforward A/B test. Assuming that I have the relevant permissions and sufficiently anonymized data, how should I select the users for this experiment?

I have two choices: 1) I could sample randomly from across the user base or 2) I could stratify my sampling accounting for the two subgroups in my users (Cities vs. Towns). If I wanted to use stratified sampling, I would sample randomly from each of the subgroups and assign half of each to the A or B treatments.

But why would I want to do this? In this case, I would stratify if I thought that there were real differences between the subgroups. For example, maybe my Cities subgroup baseline click through probability is higher than the Towns subgroup. The other possibility is that the effect of treatment B could differ between Cities and Towns. Maybe people in Cities have different aesthetics than people in Towns and really like treatment B, leading to a stronger effect of treatment B on the click through probability. I won’t be addressing that possibility in this case, but there are other cases where it may be important.

In this article I’m going to simulate a user base and conduct an experiment with a hypothesized response. First using complete random sampling (AKA simple random sampling) and then using stratified random sampling. Then I’ll examine how my inferences about the experiment change between the two sampling regimes. The baseline click through probability will differ substantially between the two subgroups, but the treatment effect will be the same for each group.

## Constructing the data

I’ll be using pandas, numpy, scipy, and statsmodels for conducting this analysis. You could do some of this with sklearn instead of statsmodels but I prefer the statistical outputs of statsmodels.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
```

The first thing I need to do is to create a user base.

```python
np.random.seed(42)
population_size = 500000
populationdf = pd.DataFrame(list(range(population_size)))
populationdf.columns = ['id']
town_size = ['city']*15 + ['town']
populationdf['town_size'] = np.random.choice(town_size,size = population_size)
```

First, for reproducibility, I set a random seed for numpy to use in its random choice algorithms. Each row in the population dataframe represents one unique user. My user base has 500,000 people, and people from cities are 15 times more abundant than people from towns. In the simulated dataset the ratio of people in cities to towns isn’t exactly 15:1 since I’m using a random process:

```
city       468520
town        31480
```

Coming from an R background, I also include an id column in this dataframe. Pandas usually accounts for this with its indexing functionality, but I like to have an invariant id number when I’m sampling from a population.

***

## Random Sampling Experiment

For this experiment, I’m interested in whether the probability of a user making a click on the app will increase after implementing a change. For example, maybe Company X is interested is changing a button that leads to a partner site in order to increase traffic.

In the first case, I’m going to randomly sample from the population as a whole, without taking into account the differences between towns and cities.

```python
experiment_size = int(1000)
completerandom_idx = populationdf.sample(int(experiment_size),random_state = 21).index
completerandom_df = populationdf.loc[completerandom_idx,:]
```

In the code above, I set a sample size of 500 users per treatment (1000 users total). I then sample 1000 users from the overall population and create a new dataframe based on those samples. Ideally, you would decide on a sample size using a power analysis. But I’ll come back to that at the end of the article.

The next step is to assign the treatments. To make things easy on myself I created a function that would do this for both the complete random sampling and the stratified random sampling.

```python
def assign_treatments(df, experiment_size, seed  = 42, level = 0):
    """
    1. Given a dataframe, add a column for treatment
    2. Randomly assign half of those rows to the 'B' treatment
    3. Get the dummy variables for the treatment column
    4. Get the dummy variables for the city column 
    5. Return a modified dataframe
    """
    workdf = df.copy()
    workdf['treatment'] = 'A'
    if level == 0:
        experiment_idx = workdf.sample(int(experiment_size/2),random_state = seed).index
    if level == 1:
        experiment_idx = list(workdf[workdf['town_size']=='town']\
                              .sample(int(experiment_size/4),random_state = seed).index)\
                         + list(workdf[workdf['town_size']=='city']\
                               .sample(int(experiment_size/4),random_state = seed).index)

    workdf.loc[experiment_idx,'treatment'] = 'B'
    dummies = pd.get_dummies(workdf['treatment'])
    citydummies = pd.get_dummies(workdf['town_size'])
    workdf = pd.concat([workdf,dummies,citydummies], axis = 1)
    return(workdf)
```
At the end of this process we get a dataframe that looks like this:
<figure>
  <img src="{{ '/assets/blog_post_figures/2019-12-12/strat.png' | relative_url }}" alt="Simulated raw data for City and Town populations">
  <figcaption class="figure-caption">Figure 1: Simulated raw data for City and Town populations.</figcaption>
</figure>
I’ve added a column for the assigned treatment as well as columns for each dummy variable. The dummy variable columns will make synthesizing the response variable easier.

How many of each city category was assigned to each treatment?
<figure>
  <img src="{{ '/assets/blog_post_figures/2019-12-12/start-1.png' | relative_url }}" alt="Bar chart of city category by treatment">
  <figcaption class="figure-caption">Figure 2: Bar chart of city category by treatment.</figcaption>
</figure>

Right off the bat we can see some concerns. The Town category is very undersampled relative to the City category (as we expected based off of their abundance in the population). More importantly, there is almost double the Town individuals in Treatment B than in Treatment A. If Town individuals are very different from City individuals, this could be a real problem. (Hint: It will be a problem here.)

***

## Random Sampling Analysis
In order to calculate an expected signal, I need to specify the baseline click through probabilities as well as the lift created by the interface change. For this example I assume that City users have a baseline click through probability of 0.3 while users in Towns have a baseline click through probability of 0.1. I then assume that users in the B group will have experience an additional 0.05 increase in their click-through probability. It’s a bit large, but useful in this toy example.

To create response data, I sampled from the binomial distribution for each group according to the probabilities that I listed in the paragraph above.

```python
completerandom_df['response'] = completerandom_df['city']*completerandom_df['A']\
                                  *np.random.binomial(n = 1, p = 0.3, size = experiment_size)\
                              + completerandom_df['town']*completerandom_df['A']\
                                  *np.random.binomial(n = 1, p = 0.1, size = experiment_size)\
                              + completerandom_df['city']*completerandom_df['B']\
                                  *np.random.binomial(n = 1, p = 0.35, size = experiment_size)\
                              + completerandom_df['town']*completerandom_df['B']\
                                  *np.random.binomial(n = 1, p = 0.15, size = experiment_size)
```
To break down what is going on in the code block above, start with the np.random.binomial() function. There I am sampling 500 times from a binomial distribution with only one trial each, where the the probability of success is any of the probabilities I just listed (0.3, 0.1, or the other two plus an additional 0.05). Here is where the dummy variables come in handy. Remember that the city column is 1 only when the user is from a city and 0 everywhere else, the same is true for the A, B, and town columns. That means we can multiply each of the four binomial response variables by each of their respective dummy variables, then add all 4 columns together to get one response variable column. This process is analogous to the one used by general linear regression models to analyze categorical data.

From here we can easily conduct a t-test to see if our treatment has an effect.

```python
responseA = completerandom_df['response'][completerandom_df['A']==1]
responseB = completerandom_df['response'][completerandom_df['B']==1]
stats.ttest_ind(responseA, responseB)
```
Running this code gives us a t-statistic of -0.90 and a p-value of 0.37. Not great. If we plot the data we can get an idea of what’s going on.

<figure>
  <img src="{{ '/assets/blog_post_figures/2019-12-12/strat-2.png' | relative_url }}" alt="Results of A/B test without accounting for stratification">
  <figcaption class="figure-caption">Figure 3: Results of A/B test without accounting for stratification.</figcaption>
</figure>
***

## Stratified Sampling Experiment & Analysis
Now instead of conducting simple random sampling, let’s use stratified random sampling and see how the analytical results change.

So how do we conduct stratified random sampling? Turns out this is pretty similar to the original complete random sampling.

Here’s the steps:

Define the subpopulations you want to sample from
From each subpopulation conduct complete random sampling
The way that I like to do this protocol is by creating k lists of indices, one for each of the subpopulations I want to sample. Then I concatenate them into one long list, and use that new index list to extract the data from my original population. I like to do it this way so that I can keep track of how the dataframe was built, but there probably is a more efficient solution.

```python
townid = populationdf['id'][populationdf['cities']=='town']
bigcityid = populationdf['id'][populationdf['cities']=='bigCity']

stratified_idx = list(townid.sample(int(experiment_size/2),random_state = 42).index) \
                 + list(bigcityid.sample(int(experiment_size/2),random_state = 42).index)
stratified_df = populationdf.loc[stratified_idx,:]
```

At this point, I calculate the response variable in the same way as I did above, then run the

The code for the t-test remains the same:
```python
responseA = stratified_df['response'][stratified_df['A']==1]
responseB = stratified_df['response'][stratified_df['B']==1]
stats.ttest_ind(responseA, responseB)
```
The output here is a t-value of 2.55 and a p-value of 0.011. Much better than the original complete random sampling regime.

I could conduct a t-test on this data again, just as I did above. However, now that I’ve made sure to sample from both subpopulations I can use an ANOVA to account for some of the variance and boost the signal.

But what happens when we use the extra information of the subpopulation and run a two-way ANOVA on the data?

```
model = ols('response ~ treatment + cities', data = stratified_df).fit()
aov_table = anova_lm(model, typ=1)
print(aov_table)
            df   sum_sq  mean_sq      F     PR(>F)
treatment    1    1.16      1.16   6.89     0.0090
town_size    1   10.82     10.82  64.47   2.75e-15
Residual   997  167.27      0.17    NaN        NaN
```

The two-way ANOVA is estimating how much each of the variables (treatment and response) contributes to the total error of the response. You can see this output in the sum_sq term in the second column of the table. Based off of this it looks like the cities variable contributes about 9.4X more variance than the treatment variable, which makes sense based on how I constructed the response variable. Also, in this case I’m not getting a much lower p-value for the treatment effect than from a traditional t-test. That’s mainly because the effect of treatment is the same between the two cities. When the effect of treatment is the same across sub groups, whether or not a two-way ANOVA outperforms a simple t-test depends on the exact distribution of the sampled data.

If we look at the plot, we can see why an ANOVA performs so well.

<figure>
  <img src="{{ '/assets/blog_post_figures/2019-12-12/strat-3.png' | relative_url }}" alt="ANOVA results by population type">
  <figcaption class="figure-caption">Figure 4: ANOVA results by population type.</figcaption>
</figure>

Because we increased the power of our analysis using the stratified random sampling, our initial sample size calculations would be different between a stratified random sampling and a complete random sampling procedure. I will not go into this in depth here, but essentially our estimates of the variance of each population changes when we go from complete random sampling to stratified random sampling.

***

## Conclusions
There are two main takeaways from this article. First, consider conducting stratified random sampling when the signal could be very different between subpopulations. Second, when you use stratified random sampling to conduct an experiment, use an analytical method that can take into account categorical variables.

Overall, stratified random sampling increases the power of your analysis. In the initial complete random sampling experiment, the signal of the A/B test was diluted by the extra variation introduced by the unaccounted for Town subpopulation. Once I accounted for that subpopulation, then the signal of the A/B test became clear.

I hope that helps with your stratified random sampling needs! I’ve added the Jupyter notebook where I conducted this analysis and a similar R script to my GitHub [repository](https://github.com/j-sadowski/FromRtoPython). Here are some links for further reading on ANOVAs and stratified random sampling:

- There are a many articles online that go over the different types of sampling methodologies. See [here](https://medium.com/analytics-vidhya/sampling-methods-6c6e21773a1c), [here](https://towardsdatascience.com/sampling-techniques-a4e34111d808), or [here](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/sampling-in-statistics/).
- This course from Penn State goes much deeper into the statistics of stratified random sampling (This course is now unavailable JSS 2025-06-16).
- Examples of how to conduct an ANOVA on the iris dataset in R and in [Python](https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/).
- How to conduct a power analysis using traditional [methods](https://www.evanmiller.org/ab-testing/sample-size.html), and using simulated data.