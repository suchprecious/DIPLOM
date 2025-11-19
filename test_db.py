import mysql.connector
from mysql.connector import Error

def test_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password='12aaaBBB@',
            database='diplom'
        )
        
        if connection.is_connected():
            print("Успешное подключение")
            
            cursor = connection.cursor()
            

            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"Найдено таблиц: {len(tables)}")
            for table in tables:
                print(f"  - {table[0]}")
            

            cursor.execute("SELECT COUNT(*) FROM technologies")
            tech_count = cursor.fetchone()[0]
            print(f"Технологий в базе: {tech_count}")
            
            cursor.execute("SELECT COUNT(*) FROM project_characteristics")
            char_count = cursor.fetchone()[0]
            print(f"Характеристик проектов: {char_count}")
            
            cursor.close()
            connection.close()
            
    except Error as e:
        print(f"Ошибка подключения: {e}")


test_mysql_connection()