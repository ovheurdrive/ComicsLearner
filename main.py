import pythonscraper.scraper as sc
import pythonscraper.db as db
import pythonscraper.comic_vine_scraper as cvs
import sys
import os
import math
import random
import re

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "db":
            db.database_create_and_populate()
            db.check_database_urls()

        elif sys.argv[1] == "metadata":
            while True:
                try:
                    cvs.api_key = cvs.load_api_key("secrets.json")["API_KEY"]
                    comics = db.query("SELECT id, title FROM comics WHERE number_of_issues>=? ORDER BY number_of_issues DESC", (100,))
                    for comic in comics:
                        print("Processing comic {}".format(comic[1]))
                        query, year = cvs.query_builder(comic[1])
                        comic_metadata = get_metadata(query, year)
                        cvs_issues = cvs.get_volume_issues(comic_metadata["api_detail_url"])
                        db_issues = db.query("SELECT * from issues WHERE comic_id=? and publication_date is NULL", (comic[0],))
                        for issue in db_issues:
                            issue_number = db.parse_issue_number_from_link(issue[1])
                            print("Updating issue {} for comic {}".format(issue_number,comic[1]))
                            cvs_issue = cvs.filters(cvs_issues, "issue_number", [issue_number])
                            if len(cvs_issue) == 0:
                                continue
                            else:
                                cvs_issue_url = cvs_issue[0]["api_detail_url"]
                                issue_metadata = cvs.get_issue_details(cvs_issue_url)
                                db.query("UPDATE issues SET publication_date=?,comic_vine_url=? WHERE id=?", (issue_metadata["cover_date"], cvs_issue_url, issue[0]))
                    break
                except:
                    continue

        elif sys.argv[1] == "download":
            comic_ids = [ ids[0] for ids in db.query("SELECT DISTINCT comic_id from issues where publication_date is not null", () )]
            for cid in comic_ids:
                issues = db.query("SELECT * from issues where publication_date is not null and comic_id=?", (cid,))
                comic_data = db.query("SELECT * from comics where id=?", (cid,))[0]
                k = len(issues)
                while len(issues) > 0.3*k:
                    issues.remove(random.choice(issues))
                for issue in issues:
                    print("Getting pages for comic issue {}, link {}, label {}".format(issue[0], issue[1], issue[3]))
                    pages = sc.getIssuePages(issue[1])
                    for page in pages[2:-1]:
                        url = page["src"]
                        filename = page["page"]
                        year = re.findall("(\d{4})-\d{2}-\d{2}", issue[3])[0        ]
                        path = "{}/{}".format(comic_data[1], year)
                        final_filename = sc.imgDownloader(url, filename, path)
                        db.query("INSERT INTO files(filename, issue_id, label) VALUES(?,?,?)", ( "comics/{}/{}.jpg".format(path,final_filename), issue[0], year ))

def get_metadata(query, year):
    res = cvs.generic_search(query, "volume")
    fres = cvs.filters(res["results"], "resource_type", ["volume"])
    if year != None:
        temp = cvs.filters(fres, "start_year", [year])
        if len(temp) > 0:
            fres = temp
    for dic in fres:
        dic.pop("description",None)
    most_relevant = sorted(fres, key=lambda x: tryInt(x["count_of_issues"]), reverse=True)[0]
    return most_relevant

def tryInt(string):
    try:
        res = int(string)
        return res
    except ValueError:
        return math.inf

if __name__ == "__main__":
    main()