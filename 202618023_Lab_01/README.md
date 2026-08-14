# Book Data Scraping and Analysis

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Drashti Akbari |
| **Student ID** | 202618023 |
| **Assignment** | Book Data Scraping and Analysis |

---

## 1. Project Overview

This project focuses on **web scraping, data preprocessing, exploratory data analysis, visualization, and interpretation** using the Books to Scrape website.

The project uses **Scrapy** to collect book information and **Pandas, Matplotlib, Seaborn, and WordCloud** for preprocessing and analysis.

The main goal is to understand book prices, ratings, categories, stock availability, and descriptions, and to identify useful patterns from the scraped dataset.

---

## 2. Dataset

### Source
**Books to Scrape** — `books.toscrape.com`

### Scraping Details

- Data was scraped using a **Scrapy spider**.
- The spider starts from page 1.
- Pagination was followed up to **page 5**.
- The scraped data was saved as `book_data.csv`.
- The dataset contains **100 books**.

### Main Attributes

| Column | Description |
|---|---|
| `title` | Title of the book |
| `category` | Book category |
| `price` | Book price |
| `rating` | Star rating |
| `availability` | Stock/availability information |
| `product_description` | Description of the book |
| `UPC` | Unique product code |
| `number_of_reviews` | Number of reviews |
| `product_url` | URL of the book |
| `stock_count` | Stock quantity used in analysis |

---

## 3. Technologies and Libraries Used

- **Python**
- **Scrapy** – Web scraping
- **Pandas** – Data preprocessing and analysis
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **WordCloud** – Text visualization
- **Jupyter Notebook** – Development environment

---

## 4. Project Details

### Task 1 – Data Scraping
A Scrapy spider was created to collect book title, category, price, rating, availability, description, UPC, number of reviews, and product URL. The spider follows pagination from **page 1 to page 5** and saves the collected records to a CSV file.

### Task 2 – Data Preprocessing
The scraped data was cleaned by checking missing values and duplicates, removing extra spaces, handling missing values, removing duplicate UPC records, converting price and rating into suitable numeric formats, and creating additional features such as description word count and price bands.

### Visualization and Analysis
The project includes visualizations for price distribution, rating distribution, stock distribution, category patterns, price-rating relationships, average price by category, and a word cloud based on book descriptions.

### Task 4 – Insights and Interpretation
The analysis examines the relationship between price and rating, highly rated books, category representation, expensive categories, stock patterns, and books that provide better value.

---

## 5. Key Observations

1. **Price does not show a strong relationship with rating.** Expensive books are not necessarily highly rated.
2. **Books with a 5-star rating are the highest-rated books** in the dataset.
3. **Book categories are not evenly represented.** Some categories contain many more books than others.
4. **Average prices differ between categories**, with some categories being more expensive than others.
5. **Books with high ratings and relatively low prices provide better value for money.**
6. The dataset contains only **100 books from the first five pages**, so it may not represent the complete website.

---

## 6. Limitations

- Only 100 books from the first five pages were scraped.
- The dataset may not represent all books available on the website.
- Customer review text was not available, so book descriptions were used for the word cloud.
- The relationship between price and rating shows association and does not prove causation.
- Category patterns are limited to the books included in the scraped pages.

---

## 7. Conclusion

This project demonstrates a complete basic data-analysis workflow, from **web scraping and data collection** to **data preprocessing, feature creation, visualization, and interpretation**.

The analysis shows that book price is not necessarily an indicator of rating. Categories differ in representation and average price, while books combining **higher ratings with relatively lower prices** can be considered better value.

Overall, the project provides practical experience with **Scrapy and Python data-analysis libraries** for extracting and interpreting information from a real-world web dataset.

---

## 8. Project Structure

```text
Book-Data-Scraping-Analysis/
│
├── README.md
├── 23_lab(2).ipynb
├── book_data.csv
└── b_spider.py
```

---

## Author

**Drashti Akbari**  
**Student ID: 202618023**
