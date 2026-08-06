import scrapy


class BooksToScrapeSpider(scrapy.Spider):
    """Crawls books.toscrape.com and yields one record per book."""

    name = "books_to_scrape"
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]
    max_pages = 6  # enough pages to comfortably clear 100+ books
    pages_visited = 1

    def parse(self, response):
        book_links = response.css("h3 a::attr(href)").getall()
        for link in book_links:
            yield response.follow(link, callback=self.parse_book_page)

        yield from self._follow_next_page(response)

    def _follow_next_page(self, response):
        next_link = response.css("li.next a::attr(href)").get()
        if next_link and self.pages_visited < self.max_pages:
            self.pages_visited += 1
            yield response.follow(next_link, callback=self.parse)

    def parse_book_page(self, response):
        def field(xpath_expr):
            return response.xpath(xpath_expr).get()

        yield {
            "title": response.css("div.product_main h1::text").get(),
            "category": response.xpath('//ul[@class="breadcrumb"]/li[3]/a/text()').get(),
            "price": response.css("p.price_color::text").get(),
            "rating": response.css("p.star-rating::attr(class)").get(),
            "availability": field('//th[text()="Availability"]/following-sibling::td/text()'),
            "description": field('//div[@id="product_description"]/following-sibling::p/text()'),
            "upc": field('//th[text()="UPC"]/following-sibling::td/text()'),
            "reviews": field('//th[text()="Number of reviews"]/following-sibling::td/text()'),
            "url": response.url,
        }
