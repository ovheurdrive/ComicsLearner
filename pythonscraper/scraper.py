import requests
import bs4
import json


def getKeywords(file):
    with open(file, "r") as keywords:
        kjson = json.load(keywords)
    return kjson

def getRelevantComicLinks(keyword):
    print("Searching for", keyword)
    res = requests.get("http://comicextra.com/comic-search?key={}".format(keyword))
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    comic_list_raw = soup.find_all("div", {"class":"cartoon-box"})
    comic_links = [ raw_html.find("h3").find("a")["href"] for raw_html in comic_list_raw]
    return comic_links

def getComicMetadata(link):
    print("Getting metadata for ", link)
    metadata = { "issues_links" : [], "title" : "", "released": "", "alternate title" : "", "author": ""  }
    res = requests.get(link)
    if res.status_code == 200:
        soup = bs4.BeautifulSoup(res.text, "html.parser")
        metadata_raw = soup.find_all("div", { "class":"movie-info-box"})

        metadata["title"] = metadata_raw[0].find("span").contents[0]
        details = metadata_raw[0].find("dl" , { "class" : "movie-dl" })
        details_values = details.find_all("dd")

        metadata["alternate title"] = details_values[2].contents[0].replace("\n", "").rstrip()
        metadata["released"] = details_values[3].contents[0].replace("\n", "").rstrip()
        metadata["author"] = details_values[4].contents[0].replace("\n", "").rstrip()

        issues_raw = soup.find("div", { "class":"episode-list"}).find_all("a")
        metadata["issues_links"] = [ issue_raw["href"] for issue_raw in issues_raw]
    return metadata

def getIssuePages(link):
    res = requests.get("{}/full".format(link))
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    page_raw = soup.find_all("img", { "class" : "chapter_img"})
    img_raw = page_raw[0].find_all("img")
    pages = [ { "page" : raw["alt"] , "src" : raw["src"] } for raw in img_raw ]
    return pages

def imgDownloader(link, filename):
    filename = filename.replace(" ", "_").replace("/", "_")
    print("Downloading ", "{}.jpg".format(filename) )
    data = requests.get(link, stream=True).content
    with open("comics/{}.jpg".format(filename), "wb") as handler:
        handler.write(data)


def downloader():
    kjson = getKeywords("comics.json")
    for key,value in kjson:
        for keyword in value:
            links = getRelevantComicLinks(keyword)
            for comic in links:
                metadata = getComicMetadata(comic)
                for issue in metadata["issues_links"]:
                    pages = getIssuePages(issue)
                    for page in pages:
                        imgDownloader(page["src"], page["page"])


# Example
kjson = getKeywords("comics.json")
keyword = kjson["DC"][9]
links = getRelevantComicLinks(keyword)
metadata = getComicMetadata(links[8])

print(metadata["issues_links"])
pages = getIssuePages(metadata["issues_links"][0])
imgDownloader(pages[10  ]["src"], pages[10]["page"])

# DCComics = []
# for keywords in kjson["DC"]:
#     comics_link = getRelevantComicLinks(keywords)
#     for link in comics_link:
#         DCComics.append(getComicMetadata(link))
# print(DCComics)