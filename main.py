import pythonscraper.scraper as sc
import pythonscraper.db as db
import pythonscraper.comic_vine_scraper as cvs
import sys
import os

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "populate":
            db.database_create_and_populate()
            db.check_database_urls()
        if sys.argv[1] == "get_metadata":
            cvs.api_key = cvs.load_api_key("secrets.json")["API_KEY"]
            res = cvs.generic_search("Detective Comics")
            fres = cvs.filters(res["results"], "resource_type", ["volume"])
            for dic in fres:
                dic.pop("description",None)
            test_issue = sorted(cvs.get_volume_issues(fres[0]["api_detail_url"]), key=lambda x: int(x["issue_number"]))[1]["api_detail_url"]
            print(cvs.get_issue_details(test_issue)["site_detail_url"])

if __name__ == "__main__":
    main()