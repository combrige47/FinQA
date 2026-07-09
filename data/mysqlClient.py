import pymysql

class MySQLClient:

    def __init__(self):
        self.conn = pymysql.connect(
            host="124.70.51.221",
            port=3306,
            user="kang",
            password="123456",
            database="finance",
            charset="utf8mb4",
            autocommit=False
        )

        self.cursor = self.conn.cursor()

    def execute(self, sql, args=None):
        if args is None:
            self.cursor.execute(sql)
        else:
            self.cursor.execute(sql, args)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def executemany(self, sql, data):
        self.cursor.executemany(sql, data)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def get_collection_name_by_kb_id(self, kb_id:int):
        sql = """
                SELECT milvus_collection_name 
                FROM fp_knowledge_base 
                WHERE id = %s
            """
        self.execute(sql, [kb_id])
        row = self.cursor.fetchone()
        if row:
            return row[0]
        return None
