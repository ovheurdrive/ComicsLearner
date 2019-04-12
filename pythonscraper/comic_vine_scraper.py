import requests
import json
import os
from operator import itemgetter

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
api_key=""

def load_api_key(file=None):
    cwd = os.getcwd()
    secrets_file = "{}/../secrets.json".format(cwd) if file == None else file 
    print(file)
    with open(secrets_file, "r") as secrets:
        secs = json.load(secrets)
    return secs


def generic_search(query):
    print("Searchin for '{}'".format(query))
    query = query.replace(" ", "%20")
    req_url = "https://comicvine.gamespot.com/api/search?api_key={}&format=json&query={}".format(api_key, query)

    res = requests.get(req_url, headers=headers)
    if res.status_code != 200:
        raise Exception("Invalid query")
    else:
        query_result = res.json()
        return query_result

def filters(dict_array, key, values):
    return [ dic for dic in dict_array if dic[key] in values ]

def get_volume_issues(volume_url):
    print("Getting issues info for volume: {}".format(volume_url))
    req_url = "{}?api_key={}&format=json".format(volume_url, api_key)
    res = requests.get(req_url, headers=headers)
    if res.status_code != 200:
        raise Exception("Request failed")
    else:
        query_result = res.json()
        return query_result["results"]["issues"]

def get_issue_details(issue_url):
    print("Getting info for issue: {}".format(issue_url))
    req_url = "{}?api_key={}&format=json".format(issue_url,api_key)
    res = requests.get(req_url, headers=headers)
    if res.status_code != 200:
        raise Exception("Request failed")
    else:
        query_res = res.json()
        return query_res["results"]

if __name__ == "__main__":
    secrets = load_api_key()
    api_key = secrets["API_KEY"]
    res = generic_search("Action Comics")
    fres = filters(res["results"], "resource_type", ["volume"])
    for dic in fres:
        dic.pop("description",None)
    print(sorted(get_volume_issues(fres[0]["api_detail_url"]), key=lambda x: int(x["issue_number"]))[:4]) 