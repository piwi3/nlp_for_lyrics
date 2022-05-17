## Natural language processing of webscraped lyrics

- Built a __text classification model__ on __song lyrics__ with the goal to __predict an artist__ from a piece of text
- To train the model, __web scraped lyrics__ from 5+ artists from this [page](www.lyrics.com) using __Beautiful Soup__ and __regular expressions__
- Utilized __bag of words__ (and vectorization) as well as __Naive Bayes__ to implement a __classification model__ for the described task  
- As an __add-on__ (just for fun): wrote a simple __command line interface__ for downloading lyrics of any artist and creating a __fancy word cloud__

<img src="https://github.com/piwi3/nlp_for_lyrics/blob/main/code/the_rolling_stones_wrdcld.png" width="600"><br/>
_Figure 1: Tongue shaped word cloud of 100 most used words in lyrics for 'The Rolling Stones'_
