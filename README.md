# ComicsLearner

Deep Learning Project on Comics

## Folders & Files


- main.py : wrapper for pythonscraper 
```
Usage : python3 main.py [db|metadata|download]
                        db : creates and fills the databas according to comics.json queries
                        metadata : downloads metadata from comicvine for the comics in the db
                        download : downloads a pseudorandom amount of files (issues with comicvine metadata)
```
- pythonscraper : contains files for scraping comicextra.com, comicvine.gamespot.com and store comic images and their metadatas in a sqlite database
- supervised_learning : Uses fine tuning to learn Date deatures from the downloded comics
- unsupervised_learning : Uses An AutoEncoder + KMeans Clustering for classifying the comic images
- comics : folders and comic images

- comics.db : prescraped database (done with pythonscraper)
- monitor.sh : monitor the downloads in the comics folder
