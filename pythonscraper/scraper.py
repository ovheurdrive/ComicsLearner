import requests
import bs4
import json
import os
from io import BytesIO
from PIL import Image


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

        metadata["title"] = metadata_raw[0].find("span").contents[0].replace("/","_").replace(" ","_")
        details = metadata_raw[0].find("dl" , { "class" : "movie-dl" })
        details_values = details.find_all("dd")

        metadata["alternate title"] = details_values[2].contents[0].replace("\n", "").rstrip()
        metadata["released"] = details_values[3].contents[0].replace("\n", "").rstrip()
        metadata["author"] = details_values[4].contents[0].replace("\n", "").rstrip()

        issues_raw = soup.find("div", { "class":"episode-list"}).find_all("a")
        metadata["issues_links"] = [ issue_raw["href"] for issue_raw in issues_raw ]
    return metadata

def getIssuePages(link):
    res = requests.get("{}/full".format(link))
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    page_raw = soup.find_all("img", { "class" : "chapter_img"})
    img_raw = page_raw[0].find_all("img")
    pages = [ { "page" : raw["alt"] , "src" : raw["src"] } for raw in img_raw ]
    return pages

def imgDownloader(link, filename, path):
    filename = filename.replace(" ", "_").replace("/", "_")

    # Create folders for file if they don't exist
    cwd = os.getcwd()
    subpaths = path.split("/")
    recursive_path = ""
    for subpath_index in range(len(subpaths)):
        recursive_path = recursive_path + "/" + subpaths[subpath_index]
        if not(os.path.isdir("{}/comics{}".format(cwd,recursive_path))):
            print("{}/comics{} does not exists ... Creating".format(cwd,recursive_path))
            os.mkdir("{}/comics{}".format(cwd,recursive_path))

    # Download file
    print("Downloading ", "{}.jpg".format(filename))
    image_data = requests.get(link, stream=True).content
    pil_image = Image.open(BytesIO(image_data))
    width, height = pil_image.size
    pil_image = pil_image.convert('RGB')
    pil_image = pil_image.resize((width//4, height//4))

    pil_image.save("comics/{}/{}.jpg".format(path,filename), format="JPEG", quality=90)



def downloader(category):
    kjson = getKeywords("comics.json")
    for keyword in kjson[category]:
        links = getRelevantComicLinks(keyword)
        for comic in links:
            metadata = getComicMetadata(comic)
            for issue in metadata["issues_links"]:
                pages = getIssuePages(issue)
                for page in pages:
                    imgDownloader(page["src"], page["page"], "{}/{}".format(metadata["released"], metadata["title"]))


if __name__ == '__main__':
    kjson = getKeywords("comics.json")
    keyword = kjson["DC"][9]
    links = getRelevantComicLinks(keyword)
    metadata = getComicMetadata(links[8])
    print(metadata["issues_links"])
    print(metadata["issues_links"][0].split("/")[-1].split("-")[-1])
    pages = getIssuePages(metadata["issues_links"][0])
    imgDownloader(pages[10]["src"], pages[10]["page"], "{}/{}".format(metadata["released"], metadata["title"]))