import scrapy


class BooksToScrapeSpider(scrapy.Spider):
    name = "books_to_scrape"
    allowed_domains = ["books.toscrape.com"]

    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    def parse(self, response):
        # Extract links of all books on the current page
        book_links = response.css("article.product_pod h3 a::attr(href)").getall()

        for link in book_links:
            yield response.follow(link, callback=self.parse_book)

        # Follow pagination only up to page 5
        next_page = response.css("li.next a::attr(href)").get()

        if next_page:
            current_page = int(response.url.split("page-")[1].split(".")[0])

            if current_page < 5:
                yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        # Extract product information
        yield {
            "title": response.css("div.product_main h1::text").get(),

            "category": response.css(
                "ul.breadcrumb li:nth-child(3) a::text"
            ).get(),

            "price": response.css(
                "div.product_main p.price_color::text"
            ).get(),

            "rating": response.css(
                "div.product_main p.star-rating::attr(class)"
            ).get().replace("star-rating ", ""),

            "availability": response.css(
                "div.product_main p.instock.availability::text"
            ).getall()[-1].strip(),

            "product_description": response.css(
                "#product_description ~ p::text"
            ).get(),

            "UPC": response.css(
                "table.table-striped tr:nth-child(1) td::text"
            ).get(),

            "number_of_reviews": response.css(
                "table.table-striped tr:nth-child(7) td::text"
            ).get(),

            "product_url": response.url
        }
