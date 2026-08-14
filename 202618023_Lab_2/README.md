# LAB 02 – Vectorized Programming with NumPy and Data Analysis with Pandas

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Drashti Akbari |
| **Student ID** | 202618023 |
| **Assignment** | Lab 02 |
| **Notebook** | `202618023_Lab_02.ipynb` |

---

## 1. Assignment Overview

This assignment demonstrates the use of **NumPy and Pandas** for numerical computing, data analysis, data cleaning, feature creation, and visualization.

The assignment covers:
- Vectorized programming with NumPy
- Statistical and matrix operations
- Normal distribution
- Titanic dataset analysis using Pandas
- Data filtering and aggregation
- Missing-value and outlier analysis
- Feature engineering and pivot tables
- Data visualization and observations

---

## 2. Dataset

### Dataset Name
`train.csv`

### Dataset
**Titanic Passenger Dataset**

### Dataset Size
- **Rows:** 891
- **Columns:** 12

### Columns

`PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

### Important Variables

| Column | Description |
|---|---|
| `Survived` | 0 = Did not survive, 1 = Survived |
| `Pclass` | Passenger class |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Siblings/spouses aboard |
| `Parch` | Parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation |

---

## 3. Objectives

- Understand NumPy arrays and vectorized operations.
- Perform statistical and linear algebra calculations.
- Generate and visualize a normal distribution.
- Load, inspect, and analyze a real-world dataset.
- Perform filtering, grouping, and aggregation.
- Handle missing values and detect outliers.
- Create new features and pivot tables.
- Visualize relationships and summarize findings.

---

## 4. Tools and Libraries Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

---

# 5. Part A – Vectorized Programming with NumPy

## Task 1 – Arrays, Statistics, and Indexing

Generated random arrays and performed statistical calculations, array creation, indexing, slicing, reshaping, and flattening.

**Array A results:**

| Statistic | Result |
|---|---:|
| Minimum | 2 |
| Maximum | 100 |
| Median | 54.0 |
| Mean | 51.54 |
| Standard Deviation | 29.2785 |

Also used `np.arange()`, `np.zeros()`, `np.ones()`, and `np.linspace()`.

---

## Task 2 – Vectorized Arithmetic and Linear Algebra

Performed matrix addition, element-wise multiplication, matrix multiplication, transpose, determinant, inverse, and inverse verification using `np.allclose()`.

**Key result:** `A @ A⁻¹` was successfully verified as the identity matrix.

---

## Task 3 – Normal Distribution and Histogram

Generated 1,000 values from a normal distribution, calculated the sample mean and standard deviation, compared the results, and plotted a histogram.

---

# 6. Part B – Data Analysis with Pandas

## Task 4 – Load and Inspect Data

Loaded `train.csv` and used `head()`, `tail()`, `shape`, `columns`, `info()`, `describe()`, `loc`, and `iloc`.

**Dataset shape:** `891 × 12`

---

## Task 5 – Filtering and Querying

Used Boolean indexing and `query()` to answer questions about passenger age, gender, class, fare, survival, and travelling status.

| Analysis | Result |
|---|---:|
| Male passengers older than 50 | **47** |
| Female first-class passengers | **94** |
| Female first-class survival rate | **96.81%** |
| Age 20–40, fare above median, survived | **104** |
| Travelling alone, age < 30, did not survive | **141** |
| Embarked S, Pclass 2/3, fare above Southampton median | **193** |

---

## Task 6 – Groupby and Aggregation

Calculated survival rates, passenger counts, average age, and average fare using `groupby()`.

**Survival rate by Sex:**
- Female: **74.20%**
- Male: **18.89%**

**Survival rate by Pclass:**
- Pclass 1: **62.96%**
- Pclass 2: **47.28%**
- Pclass 3: **24.24%**

---

## Task 7 – Missing Values and Fare Outliers

Analyzed missing values, performed Age imputation using different methods, and detected Fare outliers using the IQR method.

**Important results:**
- Missing Age values: **177 (19.87%)**
- Missing Cabin values: **687 (77.10%)**
- Missing Embarked values: **2 (0.22%)**
- Missing Age after mean imputation: **0**
- Fare outliers: **116**

---

## Task 8 – Features and Pivot Table

Created two new features:

- `FamilySize = SibSp + Parch + 1`
- `IsAlone = 1` when `FamilySize = 1`, otherwise `0`

Created a pivot table using **Sex × Pclass** with mean survival.

**Highest survival group:** Female, Pclass 1 — **96.81%**

**Lowest survival group:** Male, Pclass 3 — **13.54%**

---

## Task 9 – Visualizations and Observations

Created:
- Correlation heatmap
- Survival rate by Sex bar chart
- Age vs Fare scatter plot

### Key Relationships

- `SibSp` and `FamilySize`: **0.89**
- `Parch` and `FamilySize`: **0.78**
- `Pclass` and `Fare`: **-0.55**
- `FamilySize` and `IsAlone`: **-0.69**
- `Pclass` and `Survived`: **-0.34**
- `Fare` and `Survived`: **0.26**

---

# 7. Key Observations

1. **Female passengers had a much higher survival rate (74.20%) than male passengers (18.89%).**
2. **Passengers in better classes generally had a higher chance of survival.**
3. **Higher fares were slightly linked with higher survival.**
4. **Most passengers paid relatively low fares, while only a few paid very high fares.**
5. **Age alone did not show a clear effect on survival.**

---

# 8. Overall Conclusion

The assignment provided practical experience with **NumPy and Pandas** for data analysis.

The Titanic dataset was successfully inspected, filtered, grouped, cleaned, and visualized. New features and a pivot table were created to understand passenger characteristics and survival patterns.

The analysis shows that **gender and passenger class were important factors associated with survival**, while fare had a smaller positive relationship with survival.

Overall, the assignment demonstrates how Python can be used to **process, analyze, visualize, and interpret real-world data**.

---

# 9. Project Structure

```text
202618023-Lab-02/
│
├── README.md
├── 202618023_Lab_02.ipynb
└── train.csv
```

---

# 10. Skills Demonstrated

- NumPy array operations
- Vectorized programming
- Statistical analysis
- Linear algebra
- Normal distribution
- Pandas DataFrame handling
- Data filtering
- `groupby()` and aggregation
- Missing-value handling
- Outlier detection
- Feature engineering
- Pivot tables
- Correlation analysis
- Data visualization
- Basic data interpretation

---

## Author

**Drashti Akbari**  
**Student ID: 202618023**
