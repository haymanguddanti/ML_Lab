import pymysql


def mysqlconnect():
    # To connect MySQL database
    conn = pymysql.connect(
        host='localhost', user='root', password="pass", db='College',
    )

    cur = conn.cursor()
    cur.execute("select @@version")
    output = cur.fetchall()
    print(output)

    # To close the connection
    conn.close()
