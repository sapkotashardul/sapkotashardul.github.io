---
layout: page
title: automated calendar scheduling
when: May 2020
importance:
---

```python
import cvxpy as cp
import numpy as np
import cvxopt
```

## Calendar Scheduling

### Scheduling for a single day

Let's consider that for any given 4 hours in a day, we want to schedule one meeting amongst 3 friends. We want to take into account each person's preferred timing while also keeping in mind their availabilities.

#### Preferences and weights and availabilities

Let's represent their preferred timings with a preference matrix, where the rows represent each person's preferred ranking while the columns represent the hours of the day (in ascending order). The users rank their meeting preferences from first to last. So for a 4 hour period, the ranking for this example is given as an integer value between 1 to 4 (where 4 is the least preferred and 1 the most), allowing for repetitions for the same rank to be assigned to more than one hour of the day. 


```python
# preference matrix for 3 friends for 4 hours of a day
pref_ranking = np.array([[1, 4, 3, 4], [4, 4, 4, 1], [4, 4, 4, 4]])
pref_ranking
```




    array([[1, 4, 3, 4],
           [4, 4, 4, 1],
           [4, 4, 4, 4]])




```python
def weigh_based_on_ranks(pref):
    weighted_pref = np.zeros(np.shape(pref)) 
    for i, row in enumerate(pref):
        for j, v in enumerate(row):
            weighted_pref[i][j] = int(len(row)-v+1)
    return weighted_pref


pref_weighted = weigh_based_on_ranks(pref_ranking)
pref_weighted
```




    array([[4., 1., 2., 1.],
           [1., 1., 1., 4.],
           [1., 1., 1., 1.]])



However, using the preference matrix as is can create some problems. Those who do not have a preference over when they want the meeting to be scheduled, or only have one strong preference may not have their preferences met. This is because, in our integer programming formulation, we want to maximize each person's preferences. Thus, we normalize the preference matrix by making each person's preferences sum to the maximum possible sum for the row (i.e. each hour slot is given a value of 4).
In addition, sometimes the presence of one person can be more important than that of the rest. This feature can be included by assigning more weight to the person's rankings.


```python
ranks = np.array([[3], [1], [2]])

def normalize_with_importance(arr, ranks):
    # creating a new array to include the normalized values
    new_arr = np.zeros(np.shape(arr)) 
    target_row_sum = len(arr[0])*len(arr[0]) #len(arr[0])*10 # maximum row sum possible 
    weights = np.array([(len(ranks)-r+1) for r in ranks]) 
    weighted_target_row_sum = [w*target_row_sum for w in weights]
    for i, row in enumerate(arr):
        for j, v in enumerate(row):
            new_arr[i][j] = v * weighted_target_row_sum[i]/np.sum(row)
    return new_arr        
```


```python
pref_norm_weighted = normalize_with_importance(pref_weighted, ranks)
pref_norm_weighted
```




    array([[ 8.        ,  2.        ,  4.        ,  2.        ],
           [ 6.85714286,  6.85714286,  6.85714286, 27.42857143],
           [ 8.        ,  8.        ,  8.        ,  8.        ]])



Now, let's represent their availabilities with an availability matrix, where the rows represent each person's availability while the columns represent the hours of the day (in ascending order). The availability is given as a binary integer (where 1 means available and 0 means not available).


```python
# availability matrix

avail = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 1]])
avail
```




    array([[1, 1, 1, 0],
           [0, 0, 0, 1],
           [1, 0, 1, 1]])




```python
# avail = np.array([[0, 0, 0, 0], [0, 0, , 1], [0, 0, 0, 0]])
# avail
```

#### Creating the solution matrix as a cvxpy object


```python
h = 4 # total number of hours
p = 3 # total number of people in the meeting
d = 1 # total number of days

# solution vector using the cvxpy variable object
y = cp.Variable((h, d), boolean=True, integer=True) 
```

#### Objective function formulation

The objective here is to maximize each person's preference subject to their availability
$$ \text{Maximize: } \sum_{p=1}^{|P|=3}\sum_{h=1}^4 R_{p,h} Y_{h,1},$$ 
where $$R$$ is the preference matrix and $$|P|$$ is the total number of people being scheduled.

#### Adding the constraints as cvxpy expressions

For this illustration, we only add 2 constraints.

1) The first constraint is that we only have one meeting scheduled based on the the availabilities and preferences of the people. This constraint is implemented by including an expression for having the solution matrix $$(Y)$$ sum equal to one. $$\sum_{i=1}^{H} Y_{i} = 1,$$
where $$H$$ is the total number of hours over which the meeting is scheduled.

2) The second constraint is that at least 2 people (as an example for this illustration)  should be available for the meeting. This constraint is implemented by taking the sum of the matrix multiplication of the availability matrix $$(X)$$ and the solution vector. 
$$\sum_{p=1}^{|P|}\sum_{h=1}^H X_{p,h} Y_{h,1} \geq MN,$$
where $$|P|$$ is the total number of people who want to take part in the meeting, $MN$ is the minimum number of people we want in the meeting, and $$H$$ is the total number of hours over which the meeting is scheduled.

Another constraint, obvious in integer programming, is that the solution matrix must be an integer.
$$Y_i \in \{0,1\}; \forall_i \in \mathbb{Z}$$


```python
# Constraints in cvxpy are stored in a list

constraints = []

# Since the constraints in python are written as cvxpy objects, the expressions 
#       for the constraints also use cvxpy specific operators and functions. 

# The @ operator in cvxpy behaves the same was as numpy.matmul but for cvxpy objects 
#     for multiplying two matrices. 

# cvxpy.sum function behaves the same was as numpy.sum but for cvxpy objects to 
#           obtain the sum of all elements of a matrix. 


# cp.sum(y) returns the sum of all the elements in the y vector
constraint1 = cp.sum(y)==1 
constraints.append(constraint1) # Constraint for only one meeting. 

constraint2 = cp.sum(avail@y) >= 1 # @ is used for matrix multiplication in python. 
constraints.append(constraint2)   


```


```python
# Objective function equation

obj = cp.sum(pref_norm_weighted@y)
```

#### Solving the integer program


```python
# objective function in cvxpy is described using the cvxpy Problem object to 
#           include the objective function and the constraint

sched = cp.Problem(cp.Maximize(obj), constraints)

sched.solve()
```




    37.42857142857143



#### Solution matrix


```python
# The solution vector, where each row represents the hour of the day 
#     (in ascending order from the top)

# 1 means the meeting will be scheduled in that hour of the day

print(y.value)
```

    [[0.]
     [0.]
     [0.]
     [1.]]


Thus, the meeting will be scheduled on the fourth hour of the day.

### Scheduling over multiple days

What if we wanted to schedule the meeting for any given 2 hours of a day over the course of 3 days?
We now want to schedule at most one meeting a day. We want to do so by taking into account each person's preferred timing while also keeping in mind their availabilities.

#### Preferences and weights and availabilities

Let's represent their preferred timings with a preference matrix, where the rows represent each person's preferred ranking while the columns represent the hours of the day (in ascending order). In this case, the columns also represent the days such that after the 2 scheduling hours, the column represents a new day. For example, if a matrix has 6 columns, with specificed 2 scheduling hours for a given day, columns 1 and 2 represent the first and second hour of the first day, columns 3 and 4 represent the first and second hour of the second day, and so on. 

The users still rank their meeting preferences from first to last. So for a 2 hour period across 3 days, the ranking for this example is given as an integer value between 1 to 6 (where 6 is the least preferred and 1 the most), allowing for repetitions for the same rank to be assigned to more than one hour across all days. 


```python
# preference matrix for 3 friends for 2 hours of 3 days
pref_ranking = np.array([[1, 6, 6, 2, 3, 4], [6, 6, 6, 6, 6, 6], [4, 5, 1, 6, 6, 6]])
pref_ranking
```




    array([[1, 6, 6, 2, 3, 4],
           [6, 6, 6, 6, 6, 6],
           [4, 5, 1, 6, 6, 6]])



We use the complements of the ranks as weights to use in the objective function


```python
pref_weighted = weigh_based_on_ranks(pref_ranking)
pref_weighted
```




    array([[6., 1., 1., 5., 4., 3.],
           [1., 1., 1., 1., 1., 1.],
           [3., 2., 6., 1., 1., 1.]])



For normalizing the preference matrix, we make each person's preferences sum to the maximum possible sum for the row (i.e. each hour is given a rank of 6). The weights are normalized across all days and not within each day to account for preferences to have the meeting in a different day. In addition, sometimes the presence of one person can be more important than that of the rest. This feature can be included by assigning more weight to the person's rankings.


```python
ranks = np.array([[3], [1], [2]])

def normalize_with_importance(arr, ranks):
    # creating a new array to include the normalized values
    new_arr = np.zeros(np.shape(arr)) 
    target_row_sum = len(arr[0])*len(arr[0]) #len(arr[0])*10 # maximum row sum possible 
    weights = np.array([(len(ranks)-r+1) for r in ranks]) 
    weighted_target_row_sum = [w*target_row_sum for w in weights]
    for i, row in enumerate(arr):
        for j, v in enumerate(row):
            new_arr[i][j] = v * weighted_target_row_sum[i]/np.sum(row)
    return new_arr        
```


```python
pref_norm_weighted = normalize_with_importance(pref_weighted, ranks)
print(pref_norm_weighted)
```

    [[10.8         1.8         1.8         9.          7.2         5.4       ]
     [18.         18.         18.         18.         18.         18.        ]
     [15.42857143 10.28571429 30.85714286  5.14285714  5.14285714  5.14285714]]


Sometimes the presence of one person can be more important than that of the rest. This feature can be included by assigning more weight to the person's rankings.

Now, let's represent their availabilities with an availability matrix, where the rows represent each person's availability while the columns represent the hours of the day (in ascending order) and also the day (with the same specification as the preference matrix). The availability is given as a binary integer (where 1 means available and 0 means not available).


```python
avail = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]])
avail
```




    array([[1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 1],
           [1, 0, 1, 0, 0, 1]])



#### Creating the solution matrix as a cvxpy object


```python
h = 2 # total number of hours
p = 3 # total number of people in the meeting
d = 3 # total number of days

# solution vector using the cvxpy variable object
y = cp.Variable((h, d), boolean=True, integer=True) 
```

#### Objective function formulation

The objective here is to maximize each person's preference subject to their availability
$$ \text{Maximize: } \sum_{d=d_c}^{ND}\bigg( \sum_{p=1}^{|P|}\sum_{h=1}^H R_{p,h} Y_{h, d} \bigg),$$ where $$R$$ is the preference matrix, $$d_c$$ is the day on which the meeting schedule is requested, and $$ND$$ is the total number of days over which the meeting is being scheduled.


```python
# Objective function equation

obj = cp.sum(pref_norm_weighted[:,0:2]@y[:,0:1]) + cp.sum(pref_norm_weighted[:,2:4]@y[:,1:2])+ cp.sum(pref_norm_weighted[:,4:6]@y[:,2:3])

```

#### Adding the constraints as cvxpy expressions

For this illustration, we add 3 constraints.

1) The first constraint is that we only have at most one meeting scheduled based on the the availabilities and preferences of the people. This constraint is implemented by including an expression for having the solution matrix $$(Y)$$ sum be at most 1. $$\sum_{h=1}^{H} Y_{h, d} \leq 1,$$
where $$d$$ is the day for which the meeting is being scheduled a $$H$$ is the total number of hours over which the meeting is scheduled.

2) The second constraint is that we have a total of 3 meetings. This constraint is implemented by including an expression for having the solution matrix $$(Y)$$  sum equal to one. 
$$\sum_{d=d_c}^{ND}\sum_{h=1}^{H} Y_{h,d} = 3,$$
where $$d_c$$ is the current day, $$N_d$$ is the total number of days over which the meeting is scheduled. and $$H$$ is the total number of hours for a given day. 

3) The third constraint is that at least 2 people (as an example for this illustration) should be available for the meeting. This constraint is implemented by taking the sum of the matrix multiplication of the availability matrix $$(X)$$ and the solution vector. 
$$\sum_{p=0}^{|P|}\sum_{h=0}^H X_{p,h} Y_{h,d} \geq {MN},$$
where $${|P|}$$ is the total number of people who want to take part in the meeting, $$MN$$ is the minimum number of people we want in the meeting, $$d$$ is the day for which the meeting is being scheduled, and $$H$$ is the total number of hours over which the meeting is scheduled.

Another constraint, obvious in integer programming, is that the solution matrix must be an integer.
$$Y_{h,d} \in \{0,1\}; \forall_{h,d} \in \mathbb{Z}$$


```python
# Constraints in cvxpy are stored in a list

constraints = []

min_people = 0
max_meetings_per_day = 1
total_meetings = 2


# Since the constraints in python are written as cvxpy objects, the expressions for 
#       the constraints also use cvxpy specific operators and functions. 

# The @ operator in cvxpy behaves the same was as numpy.matmul but for cvxpy objects 
#     for multiplying two matrices.

# cvxpy.sum function behaves the same was as numpy.sum but for cvxpy objects 
#           to obtain the sum of all elements of a matrix. 


# Since the availability matrix and the preference matrix contain the hour slots for 
#       3 days, the availability matrix and the preference matrix need to be 
#       partitioned when implementing the constraints as defined above. 
#       This partitioning is done by extracting the 2 hour columns from the 
#       availaibility matrix and the corresponding day column from the 
#       solution matrix. 


# constraint 1) for the first day
constraint1a = cp.sum(y[:,0:1]) <= max_meetings_per_day
 # constraint 1) for the second day
constraint1b = cp.sum(y[:,1:2]) <= max_meetings_per_day
# constraint 1) for the second day
constraint1c = cp.sum(y[:,2:3]) <= max_meetings_per_day 

constraints.append(constraint1a) 
constraints.append(constraint1b)
constraints.append(constraint1c)

constraint2 = cp.sum(y)==total_meetings # constraint 2
constraints.append(constraint2)

# constraint 3) for the first day
constraint3a = cp.sum(avail[:,0:2]@y[:,0:1]) >= min_people 
# constraint 3) for the second day
constraint3b = cp.sum(avail[:,2:4]@y[:,1:2]) >= min_people 
# constraint 3) for the third day
constraint3c = cp.sum(avail[:,4:6]@y[:,2:3]) >= min_people 

constraints.append(constraint3a)
constraints.append(constraint3b)
constraints.append(constraint3c)

```

#### Solving the Integer Program


```python
# objective function in cvxpy is described using the cvxpy Problem object 
#           to include the objective function and the constraint

sched = cp.Problem(cp.Maximize(obj), constraints)

sched.solve()
```




    94.88571428571429



#### Solution matrix


```python
# The solution vector, where each row represents the hour of the day 
#     (in ascending order from the top) and the column represents the day.
# 1 means the meeting will be scheduled in that hour of  the day

print(y.value)
```

    [[1. 1. 0.]
     [0. 0. 0.]]


Thus, the meetings will be scheduled on the first hours of the first and second day and the second hour of the third day.

### A more complicated scheduling problem

Here we schedule 7 meetings amongst 5 friends over the course of 7 days, each day with 12 hours.


```python
np.random.seed(0) 
pref_ranking = np.random.randint(low=1, high=85, size=(5, 84))
pref_ranking
```




    array([[45, 48, 65, 68, 68, 10, 84, 22, 37, 71, 13, 59, 66, 40, 47, 82, 38, 26, 78, 73, 10, 21, 81, 70, 80, 48, 65, 83, 50, 30, 20, 20, 15, 40, 33, 66, 10, 58, 33, 32, 75, 24, 36, 76, 56, 29, 35,  1,  1, 37, 54,  6, 39, 18, 80,  5, 43, 59, 32,  2, 66, 42, 58, 36, 12, 47, 83,  1, 15, 54, 13, 43, 76, 69,  7, 69, 48,  4, 77, 53, 79, 16, 21, 59],
           [24, 80, 14, 49, 50, 70, 42, 36, 65, 70,  1, 51, 37, 35, 49,  4, 43, 78, 22, 74,  1, 11, 44, 59, 24, 60,  3, 63, 36, 68, 83, 47, 21, 82, 51, 28, 15, 42, 59, 66, 37, 11, 44, 12,  3, 52, 81, 33, 55,  1, 39, 20, 47, 43, 57, 61, 78, 31, 25,  3,  4, 14, 41, 73, 20, 73, 27, 67, 53, 68, 62, 15,  5, 68, 12, 78, 76, 57, 17, 25, 30, 22, 26, 81],
           [61, 62, 84, 34, 33, 71, 32, 14, 72, 57, 25, 80, 42, 19, 41, 55, 80, 12, 39,  2, 45, 25, 68, 83,  4, 77, 36, 62, 70, 44, 33, 12, 11, 55, 38, 29,  3, 28, 84, 24, 54, 52, 47, 21, 54, 30, 68, 36, 40, 10, 74, 42, 24,  4, 47, 51,  4, 32, 10, 11, 28, 46, 72, 40, 62, 45, 35, 35, 34,  6, 37,  1, 76, 35, 70, 54, 81, 63,  9, 62,  2, 82, 36, 41],
           [37, 49, 26, 68, 36, 31, 30, 34, 19, 18,  3, 70, 13, 45, 67, 40, 40, 76, 23, 31, 18, 71, 72, 19, 44, 84, 50, 42, 47, 22, 74, 74, 29, 82, 59,  1, 64, 17, 37, 25, 64, 68, 52,  9, 57, 33, 20, 73, 72, 14, 59, 82, 56, 65, 76, 37, 26, 33, 43, 15, 29, 21, 83, 69, 23, 84,  8, 73, 62, 14,  6,  1,  9, 80, 80, 54, 12,  5, 40, 46, 27, 75, 53, 50],
           [52, 19, 35, 52, 31, 54, 59, 44, 56, 19, 46, 66, 71, 54, 49, 60, 81, 27, 36, 59, 50, 74, 45, 14, 71, 39, 40,  9, 14,  8, 81, 23, 80,  9,  7, 82, 72, 67, 61, 17, 57, 24, 25,  5, 50, 31, 55, 26, 21, 58, 24, 28, 30, 34, 54, 52,  8, 10, 55,  1, 84, 37, 82, 21,  4, 43, 66, 21, 37, 69, 81, 48, 11, 44, 64, 32, 21, 71, 10, 61, 36, 84, 77, 19]])



We use the complements of the ranks as weights to use in the objective function


```python
pref_weighted = weigh_based_on_ranks(pref_ranking)
pref_weighted
```




    array([[40., 37., 20., 17., 17., 75.,  1., 63., 48., 14., 72., 26., 19., 45., 38.,  3., 47., 59.,  7., 12., 75., 64.,  4., 15.,  5., 37., 20.,  2., 35., 55., 65., 65., 70., 45., 52., 19., 75., 27., 52., 53., 10., 61., 49.,  9., 29., 56., 50., 84., 84., 48., 31., 79., 46., 67.,  5., 80., 42., 26., 53., 83., 19., 43., 27., 49., 73., 38.,  2., 84., 70., 31., 72., 42.,  9., 16., 78., 16., 37., 81.,  8., 32.,  6., 69., 64., 26.],
           [61.,  5., 71., 36., 35., 15., 43., 49., 20., 15., 84., 34., 48., 50., 36., 81., 42.,  7., 63., 11., 84., 74., 41., 26., 61., 25., 82., 22., 49., 17.,  2., 38., 64.,  3., 34., 57., 70., 43., 26., 19., 48., 74., 41., 73., 82., 33.,  4., 52., 30., 84., 46., 65., 38., 42., 28., 24.,  7., 54., 60., 82., 81., 71., 44., 12., 65., 12., 58., 18., 32., 17., 23., 70., 80., 17., 73.,  7.,  9., 28., 68., 60., 55., 63., 59.,  4.],
           [24., 23.,  1., 51., 52., 14., 53., 71., 13., 28., 60.,  5., 43., 66., 44., 30.,  5., 73., 46., 83., 40., 60., 17.,  2., 81.,  8., 49., 23., 15., 41., 52., 73., 74., 30., 47., 56., 82., 57.,  1., 61., 31., 33., 38., 64., 31., 55., 17., 49., 45., 75., 11., 43., 61., 81., 38., 34., 81., 53., 75., 74., 57., 39., 13., 45., 23., 40., 50., 50., 51., 79., 48., 84.,  9., 50., 15., 31.,  4., 22., 76., 23., 83.,  3., 49., 44.],
           [48., 36., 59., 17., 49., 54., 55., 51., 66., 67., 82., 15., 72., 40., 18., 45., 45.,  9., 62., 54., 67., 14., 13., 66., 41.,  1., 35., 43., 38., 63., 11., 11., 56.,  3., 26., 84., 21., 68., 48., 60., 21., 17., 33., 76., 28., 52., 65., 12., 13., 71., 26.,  3., 29., 20.,  9., 48., 59., 52., 42., 70., 56., 64.,  2., 16., 62.,  1., 77., 12., 23., 71., 79., 84., 76.,  5.,  5., 31., 73., 80., 45., 39., 58., 10., 32., 35.],
           [33., 66., 50., 33., 54., 31., 26., 41., 29., 66., 39., 19., 14., 31., 36., 25.,  4., 58., 49., 26., 35., 11., 40., 71., 14., 46., 45., 76., 71., 77.,  4., 62.,  5., 76., 78.,  3., 13., 18., 24., 68., 28., 61., 60., 80., 35., 54., 30., 59., 64., 27., 61., 57., 55., 51., 31., 33., 77., 75., 30., 84.,  1., 48.,  3., 64., 81., 42., 19., 64., 48., 16.,  4., 37., 74., 41., 21., 53., 64., 14., 75., 24., 49.,  1.,  8., 66.]])



For normalizing the preference matrix, we make each person's preferences sum to the maximum possible sum for the row (i.e. each hour is given a rank of 6). The weights are normalized across all days and not within each day to account for preferences to have the meeting in a different day. In addition, sometimes the presence of one person can be more important than that of the rest. This feature can be included by assigning more weight to the person's rankings.


```python
ranks = np.array([[3], [1], [2], [4], [5]])

def normalize_with_importance(arr, ranks):
    # creating a new array to include the normalized values
    new_arr = np.zeros(np.shape(arr)) 
    target_row_sum = len(arr[0])*len(arr[0]) #len(arr[0])*10 # maximum row sum possible 
    weights = np.array([(len(ranks)-r+1) for r in ranks]) 
    weighted_target_row_sum = [w*target_row_sum for w in weights]
    for i, row in enumerate(arr):
        for j, v in enumerate(row):
            new_arr[i][j] = v * weighted_target_row_sum[i]/np.sum(row)
    return new_arr        
```


```python
pref_norm_weighted = normalize_with_importance(pref_weighted, ranks)
print(pref_norm_weighted)
```

    [[243.38028169 225.12676056 121.69014085 103.43661972 103.43661972 456.33802817   6.08450704 383.32394366 292.05633803  85.18309859 438.08450704 158.1971831  115.6056338  273.8028169  231.21126761  18.25352113 285.97183099 358.98591549  42.5915493   73.01408451 456.33802817 389.4084507   24.33802817  91.26760563  30.42253521 225.12676056 121.69014085  12.16901408 212.95774648 334.64788732 395.49295775 395.49295775 425.91549296 273.8028169  316.3943662  115.6056338  456.33802817 164.28169014 316.3943662  322.47887324  60.84507042 371.15492958 298.14084507  54.76056338 176.45070423 340.73239437 304.22535211 511.09859155 511.09859155 292.05633803 188.61971831 480.67605634 279.88732394 407.66197183  30.42253521 486.76056338 255.54929577 158.1971831  322.47887324 505.01408451 115.6056338  261.63380282 164.28169014 298.14084507 444.16901408 231.21126761  12.16901408 511.09859155 425.91549296 188.61971831 438.08450704 255.54929577  54.76056338  97.35211268 474.5915493   97.35211268 225.12676056 492.84507042  48.67605634 194.70422535  36.50704225 419.83098592 389.4084507  158.1971831 ]
     [591.88118812  48.51485149 688.91089109 349.30693069 339.6039604  145.54455446 417.22772277 475.44554455 194.05940594 145.54455446 815.04950495 329.9009901  465.74257426 485.14851485 349.30693069 785.94059406 407.52475248  67.92079208 611.28712871 106.73267327 815.04950495 718.01980198 397.82178218 252.27722772 591.88118812 242.57425743 795.64356436 213.46534653 475.44554455 164.95049505  19.40594059 368.71287129 620.99009901  29.10891089 329.9009901  553.06930693 679.20792079 417.22772277 252.27722772 184.35643564 465.74257426 718.01980198 397.82178218 708.31683168 795.64356436 320.1980198   38.81188119 504.55445545 291.08910891 815.04950495 446.33663366 630.69306931 368.71287129 407.52475248 271.68316832 232.87128713  67.92079208 523.96039604 582.17821782 795.64356436 785.94059406 688.91089109 426.93069307 116.43564356 630.69306931 116.43564356 562.77227723 174.65346535 310.4950495  164.95049505 223.16831683 679.20792079 776.23762376 164.95049505 708.31683168  67.92079208  87.32673267 271.68316832 659.8019802  582.17821782 533.66336634 611.28712871 572.47524752  38.81188119]
     [186.55356651 178.78050124   7.77306527 396.42632884 404.19939411 108.8229138  411.97245938 551.88763426 101.04984853 217.6458276  466.38391628  38.86532636 334.24180666 513.0223079  342.01487194 233.19195814  38.86532636 567.4337648  357.56100248 645.16441752 310.92261085 466.38391628 132.14210961  15.54613054 629.61828697  62.18452217 380.88019829 178.78050124 116.59597907 318.69567612 404.19939411 567.4337648  575.20683007 233.19195814 365.33406775 435.29165519 637.39135224 443.06472046   7.77306527 474.15698155 240.96502341 256.51115395 295.37648031 497.47617736 240.96502341 427.51858992 132.14210961 380.88019829 349.78793721 582.97989535  85.50371798 334.24180666 474.15698155 629.61828697 295.37648031 264.28421922 629.61828697 411.97245938 582.97989535 575.20683007 443.06472046 303.14954558 101.04984853 349.78793721 178.78050124 310.92261085 388.65326356 388.65326356 396.42632884 614.07215643 373.10713302 652.93748279  69.95758744 388.65326356 116.59597907 240.96502341  31.09226109 171.00743597 590.75296062 178.78050124 645.16441752  23.31919581 380.88019829 342.01487194]
     [193.81287554 145.35965665 238.22832618  68.64206009 197.85064378 218.03948498 222.07725322 205.92618026 266.49270386 270.5304721  331.09699571  60.56652361 290.7193133  161.51072961  72.67982833 181.69957082 181.69957082  36.33991416 250.3416309  218.03948498 270.5304721   56.52875536  52.49098712 266.49270386 165.54849785   4.03776824 141.32188841 173.62403433 153.43519313 254.37939914  44.41545064  44.41545064 226.11502146  12.11330472 104.98197425 339.17253219  84.79313305 274.56824034 193.81287554 242.26609442  84.79313305  68.64206009 133.24635193 306.87038627 113.05751073 209.9639485  262.45493562  48.45321888  52.49098712 286.68154506 104.98197425  12.11330472 117.09527897  80.75536481  36.33991416 193.81287554 238.22832618 209.9639485  169.58626609 282.64377682 226.11502146 258.41716738   8.07553648  64.60429185 250.3416309    4.03776824 310.90815451  48.45321888  92.86866953 286.68154506 318.98369099 339.17253219 306.87038627  20.1888412   20.1888412  125.17081545 294.75708155 323.02145923 181.69957082 157.47296137 234.19055794  40.3776824  129.20858369 141.32188841]
     [ 65.85067873 131.70135747  99.77375566  65.85067873 107.75565611  61.85972851  51.88235294  81.81447964  57.86877828 131.70135747  77.82352941  37.91402715  27.93665158  61.85972851  71.83710407  49.88687783   7.98190045 115.73755656  97.77828054  51.88235294  69.84162896  21.95022624  79.81900452 141.67873303  27.93665158  91.7918552   89.79638009 151.6561086  141.67873303 153.65158371   7.98190045 123.71945701   9.97737557 151.6561086  155.64705882   5.98642534  25.94117647  35.91855204  47.89140271 135.69230769  55.87330317 121.7239819  119.72850679 159.63800905  69.84162896 107.75565611  59.86425339 117.73303167 127.71040724  53.87782805 121.7239819  113.74208145 109.75113122 101.76923077  61.85972851  65.85067873 153.65158371 149.66063348  59.86425339 167.6199095    1.99547511  95.78280543   5.98642534 127.71040724 161.63348416  83.80995475  37.91402715 127.71040724  95.78280543  31.92760181   7.98190045  73.83257919 147.66515837  81.81447964  41.90497738 105.760181   127.71040724  27.93665158 149.66063348  47.89140271  97.77828054   1.99547511  15.9638009  131.70135747]]


Now, let's represent their availabilities with an availability matrix, where the rows represent each person's availability while the columns represent the hours of the day (in ascending order) and also the day (with the same specification as the preference matrix). The availability is given as a binary integer (where 1 means available and 0 means not available).


```python
np.random.seed(0) 
avail = np.random.randint(low=0, high=2,size=(5, 84))
print(avail)
```

    [[0 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 1 0 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0 1 1 0 0 0 1 1 0 1 0 0 1 0 1]
     [1 1 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 0 0 1 0 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 0 1 0 0 0 0 1 1 0 0 1 0 1 1 1 1 0 0 0 1 0 1 1 1 0 1 0 0 1 0 1 1 0 0]
     [1 0 1 0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 1 1 0 1 0 0 1 0 1 1 1 1 0 0 0 1 1]
     [1 0 1 1 1 1 0 0 1 1 0 0 0 1 1 0 1 1 1 1 1 0 0 0 1 0 1 0 1 1 0 0 0 1 0 0 1 1 1 1 0 1 0 0 0 0 1 1 1 0 1 0 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 0 1 1 1 1 1]
     [1 1 0 0 0 0 1 1 0 1 0 0 1 0 1 0 1 0 0 0 0 1 1 1 0 1 0 1 0 0 0 0 1 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 1 0 1 1 1 0 0 1 1 0 1 1]]


#### Creating the solution matrix as a cvxpy object


```python
h = 12 # total number of hours
p = 5 # total number of people in the meeting
d = 7 # total number of days

# solution vector using the cvxpy variable object
y = cp.Variable((h, d), boolean=True, integer=True) 
```

#### Objective function formulation

The objective here is to maximize each person's preference subject to their availability
$$ \text{Maximize: } \sum_{d=d_c}^{ND}\bigg( \sum_{p=1}^{|P|}\sum_{h=1}^H R_{p,h} Y_{h, d} \bigg),$$ where $R$ is the preference matrix, $d_c$ is the day on which the meeting schedule is requested, and $ND$ is the total number of days over which the meeting is being scheduled.


```python
off = 12 # offset for the loop
idx = 0
obj = cp.sum(0) # intializing the objective function as a cvxpy expression
for i in range(y.shape[1]):
    obj += cp.sum(pref_norm_weighted[:,idx:idx+off]@y[:,i:i+1])
    idx = idx + off
```

#### Adding the constraints as cvxpy expressions

For this illustration, we add 3 constraints.

1) The first constraint is that we only have at most one meeting scheduled based on the the availabilities and preferences of the people. This constraint is implemented by including an expression for having the solution matrix $$(Y)$$ sum be at most 1. $$\sum_{h=1}^{H} Y_{h, d} \leq 1,$$
where $$d$$ is the day for which the meeting is being scheduled a $$H$$ is the total number of hours over which the meeting is scheduled.

2) The second constraint is that we have a total of 3 meetings. This constraint is implemented by including an expression for having the solution matrix $$(Y)$$  sum equal to one. 
$$\sum_{d=d_c}^{ND}\sum_{h=1}^{H} Y_{h,d} = 3,$$
where $$d_c$$ is the current day, $$ND$$ is the total number of days over which the meeting is scheduled. and $$H$$ is the total number of hours for a given day. 

3) The third constraint is that at least 2 people (as an example for this illustration) should be available for the meeting. This constraint is implemented by taking the sum of the matrix multiplication of the availability matrix $$(X)$$ and the solution vector. 
$$\sum_{p=0}^{|P|}\sum_{h=0}^H X_{p,h} Y_{h,d} \geq {MN},$$
where $${|P|}$$ is the total number of people who want to take part in the meeting, $${MN}$$ is the minimum number of people we want in the meeting, $$d$$ is the day for which the meeting is being scheduled, and $$H$$ is the total number of hours over which the meeting is scheduled.

Another constraint, obvious in integer programming, is that the solution matrix must be an integer.
$$Y_{h,d} \in \{0,1\}; \forall_{h,d} \in \mathbb{Z}$$


```python
# Constraints in cvxpy are stored in a list

constraints = []

min_people = 4
max_meetings_per_day = 1
total_meetings = 7


# Since the constraints in python are written as cvxpy objects, the expressions for 
#       the constraints also use cvxpy specific operators and functions. 

# The @ operator in cvxpy behaves the same was as numpy.matmul but for cvxpy objects 
#     for multiplying two matrices.

# cvxpy.sum function behaves the same was as numpy.sum but for cvxpy objects 
#           to obtain the sum of all elements of a matrix. 


# Since the availability matrix and the preference matrix contain the hour slots for 
#       3 days, the availability matrix and the preference matrix need to be 
#       partitioned when implementing the constraints as defined above. 
#       This partitioning is done by extracting the 2 hour columns from the 
#       availaibility matrix and the corresponding day column from the 
#       solution matrix. 


# constraint 1 for all days

for i in range(y.shape[1]):
    constraint1 = cp.sum(y[:,i:i+1]) <= max_meetings_per_day
    constraints.append(constraint1)


constraint2 = cp.sum(y)==total_meetings # constraint 2
constraints.append(constraint2)

# constraint 3 for all days 

off = 12
idx = 0
for i in range(y.shape[1]):
    constraint3 = cp.sum(avail[:,idx:idx+off]@y[:,i:i+1]) >= min_people
    constraints.append(constraint3)
    idx = idx + off

```

#### Solving the Integer Program


```python
import time
```


```python
# objective function in cvxpy is described using the cvxpy Problem object 
#           to include the objective function and the constraint

sched = cp.Problem(cp.Maximize(obj), constraints)
start_time = time.time()
sched.solve()
print("--- %s seconds ---" % (time.time() - start_time))
```

    --- 0.04108023643493652 seconds ---


#### Solution matrix


```python
# The solution vector, where each row represents the hour of the day 
#     (in ascending order from the top) and the column represents the day.
# 1 means the meeting will be scheduled in that hour of  the day

print(y.value)
```

    [[1. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1.]]

