import sqlite3
import os
import re
import requests
import pythonscraper.scraper as sc

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

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def parse_issue_number_from_link(url):
    chapter = url.split("/")[-1]
    number = chapter.replace("chapter-", "")
    return number

def handle_special_issues(conn, special_issues, comic_id):
    print("Adding special issues for comic {}".format(comic_id))
    cur = conn.cursor()
    for issue in special_issues:
        sql =''' INSERT INTO issues(url,comic_id)
              VALUES(?,?) '''
        cur.execute(sql, (issue,comic_id))

    cur.execute("SELECT number_of_issues FROM comics WHERE id=?", (comic_id,))
    conn.commit()
    comic_num = int(cur.fetchall()[0][0])
    cur.execute("UPDATE comics SET number_of_issues=? WHERE id=?",(comic_num+len(special_issues),comic_id))
    conn.commit()

def db_insert_comic(conn, comic_metadata):
    idx = 0
    special_issues = []
    # Get number of issues
    while True:
        num = 0
        last_issue_link = comic_metadata["issues_links"][idx]
        issues_number = [ s for s in re.findall(r'\d+\.?\d*', last_issue_link.split("/")[-1])]

        if len(issues_number) != 0:
            # Edge case
            if ('.' in issues_number[-1]) or ("Annual" in last_issue_link.split("/")[-1]):
                special_issues.append(comic_metadata["issues_links"][idx])
                idx += 1
                continue
            num = int(issues_number[-1])
        # Edge case
        if num > 2500:
            special_issues.append(comic_metadata["issues_links"][idx])
            idx += 1
        else:
            break

    cur = conn.cursor()
    # Check if comic exists already
    cur.execute("SELECT * FROM comics WHERE title=?", (comic_metadata["title"],))
    exists = len(cur.fetchall()) > 0
    if exists:
        print("{} Already in the database...Skipping".format(comic_metadata["title"]))
        return
    # Insert title in database
    comic = (comic_metadata["title"], comic_metadata["author"], num+1)
    sql = ''' INSERT INTO comics(title,author,number_of_issues)
              VALUES(?,?,?) '''
    cur.execute(sql, comic)
    comic_id = cur.lastrowid
    
    # Insert issues
    for issue in range(int(num)+1):
        issue = (rreplace(last_issue_link, str(num),str(issue),1), comic_id)
        sql =''' INSERT INTO issues(url,comic_id)
              VALUES(?,?) '''
        cur = conn.cursor()
        cur.execute(sql, issue)
    
    conn.commit()

    # Add special issues
    if len(special_issues):
        print(special_issues)
        handle_special_issues(conn, special_issues, comic_id)

def query(query: str, values: tuple):
    print(query, values)
    cwd = os.getcwd()
    db_file = "{}/comics.db".format(cwd)
    conn = createConnection(db_file)
    cur = conn.cursor()
    cur.execute(query, values)
    if "SELECT" in query:
        return cur.fetchall()
    elif "INSERT" in query or "UPDATE" in query:
        conn.commit()
        return 0


def database_create_and_populate():
    cwd = os.getcwd()
    db_file = "{}/comics_labelled.db".format(cwd)
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
    sql_create_files_table = """CREATE TABLE IF NOT EXISTS files (
                                        id integer PRIMARY KEY,
                                        filename text NOT NULL,
                                        issue_id integer NOT NULL,
                                        label text NOT NULL,
                                        FOREIGN KEY (issue_id) REFERENCES issue (id)
                                    );"""
    
    if conn != None:
        create_table(conn, sql_create_comics_table)
        create_table(conn, sql_create_issues_table)
        create_table(conn, sql_create_files_table)
        kjson = sc.getKeywords("comics.json")
        for key,value in kjson.items():
            for keyword in value:
                links = sc.getRelevantComicLinks(keyword)
                for comic in links:
                    metadata = sc.getComicMetadata(comic)
                    if metadata != None:
                        db_insert_comic(conn, metadata)
    conn.close()


def check_database_urls():
    cwd = os.getcwd()
    db_file = "{}/comics.db".format(cwd)
    comic_id = 0
    while True:
        conn = createConnection(db_file)
        cur = conn.cursor()
        cur.execute("SELECT id,url,comic_id FROM issues WHERE comic_id>?", (comic_id-1,))
        try:
    
            rows = cur.fetchall()
            for row in rows:
                issue_id, issue_url, comic_id = row
                print("Checking {}".format(issue_url))
                res = requests.get(issue_url)
                if res.status_code != 200:
                    print("{} url seems dead, Deleting id {} from database".format(issue_url,issue_id))
                    cur.execute("DELETE FROM issues WHERE id=?", (issue_id,))
                    conn.commit()
                    cur.execute("SELECT number_of_issues FROM comics WHERE id=?", (comic_id,))
                    comic_num = int(cur.fetchall()[0][0])
                    print("Decreasing number of issues for comic {} by one".format(comic_id))
                    cur.execute("UPDATE comics SET number_of_issues=? WHERE id=?",(comic_num-1,comic_id))
                    conn.commit()
            break
        except KeyboardInterrupt:
            print(comic_id)
            break
        except Exception as e:
            print(e)
            continue
    conn.close()