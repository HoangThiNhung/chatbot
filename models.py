import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

import locale
locale.setlocale( locale.LC_ALL, '' )

engine = create_engine('postgresql://root:123456@localhost/chatbot_dev', echo=False)

Base = declarative_base()

# DATABASE_URL="postgresql://root:123456@localhost/chatbot_dev"
# conn=psycopg2.connect(DATABASE_URL)
from pyvi.pyvi import ViTokenizer

class Menu(Base):
    __tablename__ = 'menus'
    id = Column(Integer, primary_key=True)
    category = Column(String())
    title = Column(String())
    image = Column(String())
    link = Column(String())
    price = Column(Float())
    n_gram_search_text = Column(String())

    def __init__(self, category, title, image, link, price):
        self.category = category
        self.title = title
        self.image = image
        self.link = link
        self.price = price
        self.n_gram_search_text = self.to_search_text(category, title)

    def __repr__(self):
        return "<Menu(category='%s', title='%s', price='%s')>" % (self.category, self.title, locale.currency(self.price, grouping=True))

    def to_search_text(self, category, title):
        return ViTokenizer.tokenize(category) + ' ' +  ViTokenizer.tokenize(title)

class Promotion(Base):
    __tablename__ = 'promotions'
    id = Column(Integer, primary_key=True)
    start_date = Column(DateTime())
    end_date = Column(DateTime())
    code = Column(String())
    discount = Column(Float())
    content = Column(String())
    image = Column(String())

    def __init__(self, start_date, end_date, code, discount, content, image):
        self.start_date = start_date
        self.end_date = end_date
        self.code = code
        self.discount = discount
        self.content = content
        self.image = image

    def __repr__(self):
        return "<Promotion(code='%s', discount='%s', content='%s')>" % (self.code, "{0:.0f}%".format(self.discount), self.content)
