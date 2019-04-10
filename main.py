import pythonscraper.scraper as sc
import sqlite3
import os
import re
import requests

def createConnection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def db_insert_comic(conn, comic_metadata):
    idx = 0
    special_issues = []
    while True:
        num = 0
        last_issue_link = comic_metadata["issues_links"][idx]
        issues_number = [ s for s in re.findall(r'\d+\.?\d*', last_issue_link.split("/")[-1])]

        if len(issues_number) != 0:
            num = int(issues_number[-1])
        if num > 2000:
            special_issues.append(comic_metadata["issues_links"][idx])
            idx += 1
        else:
            break

    comic = (comic_metadata["title"], comic_metadata["author"], num+1)
    sql = ''' INSERT INTO comics(title,author,number_of_issues)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, comic)
    comic_id = cur.lastrowid
    
    for issue in range(int(num)+1):
        issue = (last_issue_link.replace(str(num),str(issue)), comic_id)
        sql =''' INSERT INTO issues(url,comic_id)
              VALUES(?,?) '''
        cur = conn.cursor()
        cur.execute(sql, issue)
    
    conn.commit()


def database_create_and_populate():
    cwd = os.getcwd()
    db_file = "{}/comics.db".format(cwd)
    if not os.path.exists(db_file):
        os.mknod(db_file)

    conn = createConnection(db_file)
    sql_create_comics_table = """CREATE TABLE IF NOT EXISTS comics (
                                        id integer PRIMARY KEY,
                                        title text NOT NULL,
                                        author text NOT NULL,
                                        number_of_issues integer NOT NULL,
                                        first_publication_date text,
                                        type text
                                    );"""
    sql_create_issues_table = """CREATE TABLE IF NOT EXISTS issues (
                                        id integer PRIMARY KEY,
                                        url text NOT NULL,
                                        comic_id integer NOT NULL,
                                        publication_date text,
                                        FOREIGN KEY (comic_id) REFERENCES comics (id)
                                    );"""
    
    if conn != None:
        create_table(conn, sql_create_comics_table)
        create_table(conn, sql_create_issues_table)
    
        kjson = sc.getKeywords("comics.json")
        for key,value in kjson.items():
            for keyword in value:
                links = sc.getRelevantComicLinks(keyword)
                for comic in links:
                    metadata = sc.getComicMetadata(comic)
                    db_insert_comic(conn, metadata)

        # kjson = sc.getKeywords("comics.json")
        # keyword = kjson["DC"][9]
        # links = sc.getRelevantComicLinks(keyword)
        # metadata = sc.getComicMetadata(links[8])
        # db_insert_comic(conn, metadata)
    conn.close()


def check_database_urls():
    cwd = os.getcwd()
    db_file = "{}/comics.db".format(cwd)
    conn = createConnection(db_file)
    cur = conn.cursor()
    cur.execute("SELECT id,url,comic_id FROM issues")
    
    rows = cur.fetchall()
    for row in rows:
        issue_id, issue_url, comic_id = row
        res = requests.get(issue_url)
        if res.status_code != 200:
            print("{} url seems dead, Deleting id {} from database".format(issue_url,issue_id))
            cur.execute("DELETE FROM issues WHERE id=?", (issue_id,))
            conn.commit()
            cur.execute("SELECT number_of_issues FROM comics WHERE id=?", (comic_id,))
            comic_num = int(cur.fetchall()[0][0])
            print("Decresing number of issues for comic {} by one".format(comic_id))
            cur.execute("UPDATE comics SET number_of_issues=? WHERE id=?",(comic_num-1,comic_id))
            conn.commit()
    conn.close()


def main():
    database_create_and_populate()
    check_database_urls()

main()